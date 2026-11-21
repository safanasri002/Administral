"""Tests for local inconsistency detection (ai.inconsistency).

No database, no OCR engine and no Ollama needed: documents are built in memory
with PyMuPDF/Pillow, and the local agent's HTTP call is replaced.
"""

from __future__ import annotations

import io

import fitz
import pytest
from PIL import Image

from app.core.config import settings
from app.modules.ai.inconsistency import local_agent
from app.modules.ai.inconsistency.consistency import AnalysedDocument, DeclaredProfile, run_rules
from app.modules.ai.inconsistency.information_extraction import extract_facts, fold, parse_amount
from app.modules.ai.inconsistency.metadata import extract_metadata
from app.modules.ai.inconsistency.service import analyse_document, compare
from app.modules.ai.inconsistency.text_extraction import extract_local_text

BAIL_COLOC = """CONTRAT DE LOCATION — BAIL DE COLOCATION
Entre M. Jean Martin, bailleur, demeurant 3 rue de Rivoli 75001 Paris
Et Mme Camille Durand, colocataire, ci-après le preneur.
Désignation : logement situé 12 rue des Lilas 69003 Lyon
Surface habitable : 42 m²
Loyer mensuel hors charges : 650,00 €
Charges mensuelles : 50,00 €
La sous-location est interdite sans l'accord écrit du bailleur.
Fait à Lyon, le 15 janvier 2025.
""" + "Clause générale du contrat de location. " * 5

BAIL_SIMPLE = """CONTRAT DE LOCATION
Le bailleur loue à Mme Camille Durand, locataire, le logement situé
12 rue des Lilas 69003 Lyon. Loyer mensuel hors charges : 650 €.
La sous-location est interdite.
""" + "Clause générale du contrat de location. " * 5


def _profile(**overrides) -> DeclaredProfile:
    base = dict(
        prenom="Camille", nom="Durand", situation_logement="locataire", type_location="vide",
        adresse="12 rue des Lilas", code_postal="69003", ville="Lyon",
        loyer_mensuel=650.0, surface_m2=42.0,
    )
    base.update(overrides)
    return DeclaredProfile(**base)


def _doc(text: str, name: str = "bail.pdf") -> AnalysedDocument:
    return AnalysedDocument(facts=extract_facts(text, name, method="texte_natif"), text=text)


@pytest.fixture(autouse=True)
def _no_agent(monkeypatch):
    """The agent is off unless a test turns it on — no test reaches the network."""
    monkeypatch.setattr(settings, "local_llm_enabled", False)


# --- Information extraction ------------------------------------------------

def test_fold_keeps_positions() -> None:
    text = "Hébergée à Évry"
    assert fold(text) == "hebergee a evry"
    assert len(fold(text)) == len(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("650,00", 650.0), ("1 250,50", 1250.5), ("1.250,50", 1250.5), ("650.00", 650.0), ("1.250", 1250.0)],
)
def test_parse_amount(raw: str, expected: float) -> None:
    assert parse_amount(raw) == expected


def test_lease_facts_are_extracted_with_their_evidence() -> None:
    facts = extract_facts(BAIL_COLOC, "bail.pdf")

    assert facts.type_document and facts.type_document.valeur == "bail"
    assert facts.statut_occupation and facts.statut_occupation.valeur == "colocataire"
    assert "colocataire" in fold(facts.statut_occupation.preuve or "") or "colocation" in fold(
        facts.statut_occupation.preuve or ""
    )
    assert facts.loyer and facts.loyer.valeur == "650.00"
    assert facts.surface and facts.surface.valeur == "42"
    assert {f.valeur.split()[0] for f in facts.codes_postaux} == {"75001", "69003"}
    assert "2025-01-15" in {d.valeur for d in facts.dates}
    assert any("Durand" in n.valeur for n in facts.noms)


def test_forbidden_sublet_clause_does_not_make_a_subtenant() -> None:
    """Nearly every lease forbids sub-letting; that says nothing about the tenant."""
    facts = extract_facts(BAIL_SIMPLE, "bail.pdf")
    assert facts.statut_occupation and facts.statut_occupation.valeur == "locataire"


def test_amounts_are_not_postcodes() -> None:
    facts = extract_facts("Revenu fiscal de référence : 15000 euros", "avis.pdf")
    assert facts.codes_postaux == []


# --- Metadata and text -------------------------------------------------------

def _pdf(text: str, author: str = "Agence Lilas") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    doc.set_metadata({"author": author, "title": "Bail", "producer": "Writer"})
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_metadata_and_native_text_without_ocr(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ocr_backend", "none")
    data = _pdf("Contrat de location - Mme Camille Durand, locataire, 69003 Lyon")

    meta = extract_metadata(data, "application/pdf")
    assert meta.auteur == "Agence Lilas" and meta.titre == "Bail" and meta.nb_pages == 1

    local = extract_local_text(data, "application/pdf")
    assert local.method == "texte_natif" and local.ocr_pages == []
    assert "Durand" in local.text


def test_image_without_local_ocr_degrades_without_cloud(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ocr_backend", "none")
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")

    local = extract_local_text(buffer.getvalue(), "image/png")
    assert local.text == "" and local.method == "aucune"
    assert local.warnings and "OCR local" in local.warnings[0]


def test_image_exif_gps_is_read() -> None:
    image = Image.new("RGB", (10, 10), "white")
    exif = image.getexif()
    exif[0x010F] = "Canon"
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (36.0, 48.0, 0.0), 3: "E", 4: (10.0, 10.0, 48.0)})
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    meta = extract_metadata(buffer.getvalue(), "image/jpeg")
    assert meta.appareil == "Canon"
    assert meta.gps and round(meta.gps.latitude, 2) == 36.8 and round(meta.gps.longitude, 2) == 10.18


def test_analyse_document_stores_facts_and_text(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ocr_backend", "none")
    stored = analyse_document(_pdf("Quittance de loyer - Mme Camille Durand"), "application/pdf", "q.pdf")
    assert stored["facts"]["type_document"]["valeur"] == "quittance_loyer"
    assert stored["facts"]["metadonnees"]["auteur"] == "Agence Lilas"
    assert "Durand" in stored["text"]


# --- Consistency rules ---------------------------------------------------------

def test_profile_tenant_vs_document_cotenant_is_reported_with_proof() -> None:
    report = compare("D1", _profile(), [_doc(BAIL_COLOC)])

    status = [i for i in report.incoherences if i.champ == "statut_occupation"]
    assert len(status) == 1
    finding = status[0]
    assert finding.portee == "profil_document" and finding.gravite == "moyenne"
    assert finding.valeur_profil == "Locataire"
    assert finding.valeurs_documents[0].valeur == "Colocataire"
    assert finding.statut == "a_verifier"
    assert "⚠️ Incohérence potentielle" in report.rapport_texte
    assert "Preuve détectée" in report.rapport_texte
    assert "À vérifier par l'utilisateur" in report.rapport_texte


def test_consistent_dossier_yields_no_finding() -> None:
    report = compare("D1", _profile(), [_doc(BAIL_SIMPLE)])
    assert report.incoherences == []
    assert report.rapport_texte.startswith("✅")


def test_address_name_and_rent_mismatches() -> None:
    profile = _profile(code_postal="75011", ville="Paris", nom="Lefebvre", loyer_mensuel=800.0)
    result = run_rules(profile, [_doc(BAIL_SIMPLE)])
    fields = {i.champ for i in result.inconsistencies}
    assert {"adresse", "identite", "loyer"} <= fields


def test_name_tolerates_one_ocr_slip() -> None:
    result = run_rules(_profile(nom="Durand"), [_doc(BAIL_SIMPLE.replace("Durand", "Duraud"))])
    assert not [i for i in result.inconsistencies if i.champ == "identite"]


def test_documents_contradicting_each_other() -> None:
    attestation = (
        "ATTESTATION D'HÉBERGEMENT\nJe certifie sur l'honneur héberger Mme Camille Durand "
        "à titre gratuit au 12 rue des Lilas 69003 Lyon.\n" + "Texte. " * 30
    )
    result = run_rules(
        _profile(), [_doc(BAIL_SIMPLE), _doc(attestation, "attestation.pdf")]
    )
    doc_doc = [i for i in result.inconsistencies if i.portee == "document_document"]
    assert doc_doc and doc_doc[0].champ == "statut_occupation" and doc_doc[0].gravite == "elevee"


def test_unreadable_document_is_reported_as_unverifiable() -> None:
    result = run_rules(_profile(), [_doc("", "scan.png")])
    assert result.inconsistencies == []
    assert any("scan.png" in p for p in result.unverifiable)


# --- Local agent -----------------------------------------------------------------

@pytest.mark.parametrize(
    "url", ["http://localhost:11434/api/chat", "http://127.0.0.1:11434", "http://192.168.1.20:11434", "http://ollama:11434"]
)
def test_agent_accepts_local_hosts(url: str) -> None:
    local_agent._assert_local(url)


@pytest.mark.parametrize("url", ["https://api.mistral.ai/v1/chat", "http://8.8.8.8", "http://[2a00:1450::1]:11434"])
def test_agent_refuses_remote_hosts(url: str) -> None:
    with pytest.raises(local_agent.AgentRefused):
        local_agent._assert_local(url)


def test_agent_opinion_downgrades_but_never_deletes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "local_llm_enabled", True)
    monkeypatch.setattr(
        local_agent,
        "_chat",
        lambda payload: {
            "avis": [{"id": "INC-01", "verdict": "infirme", "explication": "Bail de colocation au nom de l'usager."}],
            "statuts": [],
        },
    )
    report = compare("D1", _profile(), [_doc(BAIL_COLOC)])

    finding = next(i for i in report.incoherences if i.champ == "statut_occupation")
    assert finding.avis_agent == "infirme" and finding.gravite == "faible"
    assert report.agent_ia.utilise is True


def test_agent_status_needs_a_verbatim_quote(monkeypatch) -> None:
    lease = "CONTRAT DE LOCATION\nLe logement est partagé : Mme Camille Durand occupe une chambre.\n" + "Clause. " * 40
    monkeypatch.setattr(settings, "local_llm_enabled", True)

    def answer(quote: str):
        return lambda payload: {
            "avis": [],
            "statuts": [{"fichier": "bail.pdf", "statut": "colocataire", "citation": quote}],
        }

    monkeypatch.setattr(local_agent, "_chat", answer("Mme Camille Durand est colocataire du logement"))
    invented = compare("D1", _profile(), [_doc(lease)])
    assert not [i for i in invented.incoherences if i.source == "agent_ia"]

    monkeypatch.setattr(local_agent, "_chat", answer("Le logement est partagé : Mme Camille Durand occupe une chambre."))
    grounded = compare("D1", _profile(), [_doc(lease)])
    assert [i for i in grounded.incoherences if i.source == "agent_ia"]


def test_agent_unavailable_keeps_rule_findings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "local_llm_enabled", True)

    def boom(payload):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(local_agent, "_chat", boom)
    report = compare("D1", _profile(), [_doc(BAIL_COLOC)])
    assert report.agent_ia.disponible is False and report.agent_ia.note
    assert any(i.champ == "statut_occupation" for i in report.incoherences)


# --- Verdict frozen at submission -------------------------------------------------

def _verdict_for(monkeypatch, docs):
    from types import SimpleNamespace

    from app.modules.ai.inconsistency import service as svc

    analysed = iter(docs)
    monkeypatch.setattr(svc, "_ensure_analysis", lambda _db, _doc: next(analysed))
    citizen = SimpleNamespace(
        first_name="Camille", last_name="Durand",
        profile_data={"prenom": "Camille", "nom": "Durand", "situation_logement": "locataire",
                      "type_location": "vide", "code_postal": "69003", "ville": "Lyon",
                      "adresse": "12 rue des Lilas", "loyer_mensuel": 650, "surface_m2": 42},
    )
    application = SimpleNamespace(documents=[object()] * len(docs))
    return svc.submission_verdict(None, citizen, application)


def test_verdict_counts_unresolved_discrepancies_only(monkeypatch) -> None:
    verdict = _verdict_for(monkeypatch, [_doc(BAIL_COLOC)])
    assert verdict.statut == "ecarts_non_resolus"
    assert verdict.nb_ecarts == verdict.nb_elevee + verdict.nb_moyenne + verdict.nb_faible >= 1
    # Minimisation: status and counts — no discrepancy, no evidence, no text.
    stored = verdict.model_dump(mode="json")
    assert set(stored) == {
        "statut", "nb_ecarts", "nb_elevee", "nb_moyenne", "nb_faible", "nb_documents",
        "nb_points_non_verifiables", "methode", "version", "calcule_le",
    }


def test_verdict_coherent_and_non_verifiable(monkeypatch) -> None:
    assert _verdict_for(monkeypatch, [_doc(BAIL_SIMPLE)]).statut == "coherent"
    assert _verdict_for(monkeypatch, [_doc("", "scan.png")]).statut == "non_verifiable"

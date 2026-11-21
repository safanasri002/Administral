"""Replis quand l'API Mistral refuse les appels (quota à 0, plan inactif, réseau).

Deux garanties : une pièce déposée reste classée d'après son intitulé, et
l'assistant rédige avec le modèle local au lieu de répondre « indisponible » —
dans les deux cas avec exactement le même contrat de réponse.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.modules.chatbot.rag import llm_client
from app.modules.citizen.checklist_rules import generate_personalized_checklist
from app.modules.citizen.classification import classify_document
from app.modules.profiling.schemas.profil import ProfilPartiel

CHECKLIST = generate_personalized_checklist(
    ProfilPartiel(situation_logement="locataire", type_location="vide", statut_professionnel="salarie")
)


@pytest.fixture
def sans_cle(monkeypatch):
    monkeypatch.setattr(settings, "mistral_api_key", None)


@pytest.mark.parametrize(
    ("texte", "piece"),
    [
        ("CONTRAT DE LOCATION\nLe bailleur délivre quittance. La sous-location est interdite.", "contrat_location"),
        ("QUITTANCE DE LOYER\nMme Camille Durand, locataire", "attestation_loyer"),
        ("AVIS D'IMPÔT SUR LES REVENUS 2024", "avis_imposition"),
        ("BULLETIN DE PAIE\nNet à payer : 1 400,00", "bulletins_salaire_3_derniers_mois"),
        ("CARTE NATIONALE D'IDENTITÉ\nNom : DURAND", "piece_identite"),
        ("RELEVÉ D'IDENTITÉ BANCAIRE\nIBAN FR76", "releve_identite_bancaire"),
    ],
)
def test_piece_classee_par_son_intitule(sans_cle, texte: str, piece: str) -> None:
    resultat = classify_document(texte, CHECKLIST)
    assert resultat.decision == "match"
    assert resultat.matched_checklist_document_id == piece
    assert resultat.confidence >= 0.5 and resultat.evidence and resultat.error is None


def test_specimen_et_piece_non_attendue_ne_cochent_rien(sans_cle) -> None:
    specimen = classify_document("CONTRAT DE LOCATION\nSPECIMEN - exemple de bail", CHECKLIST)
    assert specimen.decision == "example_or_template" and specimen.matched_checklist_document_id is None
    hors_dossier = classify_document("LIVRET DE FAMILLE", CHECKLIST)
    assert hors_dossier.decision == "not_expected" and hors_dossier.matched_checklist_document_id is None


def test_api_en_panne_bascule_sur_les_regles(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(settings, "mistral_api_key", "cle-refusee")

    def refuse(*_a, **_k):
        raise httpx.HTTPStatusError("429", request=None, response=None)

    monkeypatch.setattr(httpx, "post", refuse)
    assert classify_document("QUITTANCE DE LOYER", CHECKLIST).decision == "match"


# --- Assistant : rédaction par le modèle local -----------------------------------

class _ClientEnPanne:
    class chat:  # noqa: N801 — imite `client.chat.complete`
        @staticmethod
        def complete(**_kwargs):
            raise RuntimeError("Status 429. Rate limit exceeded")


def test_redaction_par_le_modele_local(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "get_mistral_client", lambda: _ClientEnPanne())
    monkeypatch.setattr(settings, "chatbot_local_fallback", True)
    monkeypatch.setattr(settings, "local_llm_enabled", True)
    recu = {}

    def faux_local(messages, *, json_schema=None, temperature=0.1, **_k):
        recu["schema"] = json_schema
        return '{"type": "answer", "text": "Réponse locale.", "repondu": true}'

    monkeypatch.setattr(llm_client.local_llm, "chat", faux_local)
    resultat = llm_client.call_llm_structured([{"role": "user", "content": "x"}])
    assert resultat["text"] == "Réponse locale." and resultat["repondu"] is True
    assert recu["schema"]["required"] == ["type", "text", "repondu"]


def test_sans_repli_la_panne_remonte(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "get_mistral_client", lambda: _ClientEnPanne())
    monkeypatch.setattr(llm_client.local_llm, "chat", lambda *_a, **_k: pytest.fail("ne doit pas être appelé"))
    with pytest.raises(RuntimeError, match="429"):
        llm_client.call_llm([{"role": "user", "content": "x"}], repli_local=False)


def test_modele_local_en_panne_remonte_la_cause_d_origine(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "get_mistral_client", lambda: _ClientEnPanne())
    monkeypatch.setattr(settings, "chatbot_local_fallback", True)
    monkeypatch.setattr(settings, "local_llm_enabled", True)

    def local_absent(*_a, **_k):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(llm_client.local_llm, "chat", local_absent)
    with pytest.raises(RuntimeError, match="429"):
        llm_client.call_llm([{"role": "user", "content": "x"}])


def test_nombre_invente_par_le_modele_local_ecarte_la_reponse(monkeypatch) -> None:
    """Le modèle local a fourni un numéro de téléphone absent de tous les extraits :
    une telle réponse devient « non répondu », jamais un texte affiché au citoyen."""
    monkeypatch.setattr(llm_client, "get_mistral_client", lambda: _ClientEnPanne())
    monkeypatch.setattr(settings, "chatbot_local_fallback", True)
    monkeypatch.setattr(settings, "local_llm_enabled", True)
    messages = [{"role": "user", "content": "Extrait : le délai moyen est de 36,5 jours (1,2 mois)."}]

    monkeypatch.setattr(
        llm_client.local_llm, "chat",
        lambda *_a, **_k: '{"type": "answer", "text": "Appelez le 09 69 39 39 39.", "repondu": true}',
    )
    assert llm_client.call_llm_structured(messages)["repondu"] is False

    monkeypatch.setattr(
        llm_client.local_llm, "chat",
        lambda *_a, **_k: '{"type": "answer", "text": "1. Le délai est de 36,5 jours, soit 1,2 mois.", "repondu": true}',
    )
    assert llm_client.call_llm_structured(messages)["repondu"] is True

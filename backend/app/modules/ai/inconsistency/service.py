"""Orchestration: per-document analysis at upload, report on demand.

    analyse_document()  — upload time, once per file:
        metadata → text (native / local OCR) → facts  ⇒  ApplicationDocument.local_analysis

    build_report()      — on request:
        profile + stored facts → rules → local agent (ambiguous only) → report

Facts are stored, not the conclusions: the profile can change after an upload,
so the comparison is always recomputed against the current profile — cheap,
since no OCR is re-run. A document analysed before this module existed (or by
an older rule version) is analysed lazily from its stored file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.modules.agent.models import Citizen
from app.modules.ai.inconsistency import local_agent
from app.modules.ai.inconsistency.consistency import (
    AnalysedDocument,
    DeclaredProfile,
    run_rules,
)
from app.modules.ai.inconsistency.information_extraction import extract_facts
from app.modules.ai.inconsistency.metadata import extract_metadata
from app.modules.ai.inconsistency.schemas import (
    FACTS_VERSION,
    DocumentFacts,
    Inconsistency,
    InconsistencyReport,
    SubmissionVerdict,
)
from app.modules.ai.inconsistency.text_extraction import LocalText, extract_local_text
from app.modules.citizen.models import Application, ApplicationDocument
from app.modules.profiling.schemas.profil import ProfilPartiel

#: Full text is kept server-side (PostgreSQL) for report-time checks, capped so a
#: 200-page PDF does not bloat the row. It is never returned by the API.
_STORED_TEXT_CHARS = 30_000

_SEVERITY_ORDER = {"elevee": 0, "moyenne": 1, "faible": 2}
_SEVERITY_LABEL = {"elevee": "Élevée", "moyenne": "Moyenne", "faible": "Faible"}


def analyse_document(
    data: bytes,
    mime_type: str,
    file_name: str,
    *,
    path: str | None = None,
    local_text: LocalText | None = None,
    external_text: str | None = None,
) -> dict:
    """Metadata + text + facts for one file, as the JSONB stored on the document.

    `local_text` reuses an extraction already done by the upload flow so OCR
    never runs twice; `external_text` is text obtained another way (the
    optional cloud OCR fallback), recorded as such.
    """
    metadata = extract_metadata(data, mime_type, path)
    if local_text is None and external_text is None:
        local_text = extract_local_text(data, mime_type)
    if external_text:
        text, method, ocr_pages = external_text, "fourni", []
        warnings = list(local_text.warnings) if local_text else []
    else:
        assert local_text is not None
        text, method, ocr_pages, warnings = (
            local_text.text, local_text.method, local_text.ocr_pages, local_text.warnings
        )
    facts = extract_facts(
        text, file_name, method=method, ocr_pages=ocr_pages, metadata=metadata, warnings=warnings
    )
    return {
        "facts": facts.model_dump(mode="json"),
        "text": text[:_STORED_TEXT_CHARS],
    }


def _ensure_analysis(db: Session, document: ApplicationDocument) -> AnalysedDocument | None:
    stored = document.local_analysis or {}
    if (stored.get("facts") or {}).get("version") != FACTS_VERSION:
        file = Path(document.stored_path)
        if not file.is_file():
            return None
        stored = analyse_document(
            file.read_bytes(), document.mime_type, document.file_name, path=str(file)
        )
        document.local_analysis = stored
        db.flush()
    facts = DocumentFacts.model_validate(stored["facts"])
    # The citizen sees their own file name, not the one stored at upload time.
    facts.fichier = document.file_name
    return AnalysedDocument(facts=facts, text=stored.get("text") or "")


def declared_profile(citizen: Citizen) -> DeclaredProfile:
    profil = ProfilPartiel.model_validate(citizen.profile_data or {})
    return DeclaredProfile(
        prenom=profil.prenom or citizen.first_name,
        nom=profil.nom or citizen.last_name,
        situation_logement=profil.situation_logement.value if profil.situation_logement else None,
        type_location=profil.type_location.value if profil.type_location else None,
        adresse=profil.adresse,
        code_postal=profil.code_postal,
        ville=profil.ville,
        loyer_mensuel=profil.loyer_mensuel,
        surface_m2=profil.surface_m2,
    )


def render_text(report: InconsistencyReport) -> str:
    """The report as plain text — the format the citizen-facing summary uses."""
    if not report.incoherences:
        lines = ["✅ Aucune incohérence détectée entre votre profil et vos documents."]
    else:
        lines = []
        for inc in report.incoherences:
            lines.append(f"⚠️ Incohérence potentielle — {inc.champ} ({_SEVERITY_LABEL[inc.gravite]})")
            if inc.valeur_profil:
                lines.append(f"Profil : {inc.valeur_profil}")
            for v in inc.valeurs_documents:
                lines.append(f"Document ({v.fichier}) : {v.valeur or 'non trouvé'}")
            proofs = [v.preuve for v in inc.valeurs_documents if v.preuve]
            if proofs:
                lines.append("Preuve détectée :")
                lines.extend(f'"{p}"' for p in proofs[:3])
            if inc.explication_agent:
                lines.append(f"Avis de l'agent IA ({inc.avis_agent}) : {inc.explication_agent}")
            lines.append("→ À vérifier par l'utilisateur")
            lines.append("")
    if report.points_non_verifiables:
        lines.append("Points non vérifiables :")
        lines.extend(f"- {p}" for p in report.points_non_verifiables)
    return "\n".join(lines).strip()


def compare(
    dossier_id: str, profile: DeclaredProfile, docs: list[AnalysedDocument]
) -> InconsistencyReport:
    """Rules, then the local agent on what the rules left ambiguous. Pure of I/O
    except the local LLM call, so it is testable without a database."""
    engine = run_rules(profile, docs)
    agent = local_agent.review(profile, docs, engine)
    findings: list[Inconsistency] = engine.inconsistencies + agent.new_findings
    findings.sort(key=lambda i: (_SEVERITY_ORDER[i.gravite], i.id))
    report = InconsistencyReport(
        dossier_id=dossier_id,
        genere_le=datetime.now(UTC),
        nb_documents_analyses=len(docs),
        incoherences=findings,
        points_non_verifiables=engine.unverifiable,
        agent_ia=agent.status,
        rapport_texte="",
    )
    report.rapport_texte = render_text(report)
    return report


def build_report(db: Session, citizen: Citizen) -> InconsistencyReport:
    """Compare the citizen's current profile with every document of their dossier."""
    from app.modules.citizen import dossier  # local: citizen imports this module

    application = dossier.ensure_application_for_citizen(db, citizen)
    docs: list[AnalysedDocument] = []
    missing: list[str] = []
    for document in application.documents:
        try:
            analysed = _ensure_analysis(db, document)
        except Exception as exc:  # noqa: BLE001 — one bad file must not sink the report
            logger.warn("Analyse locale impossible", {"document": document.id, "error": str(exc)})
            analysed = None
        if analysed is None:
            missing.append(f"{document.file_name} : fichier introuvable ou illisible sur le serveur.")
        else:
            docs.append(analysed)
    db.commit()

    report = compare(application.id, declared_profile(citizen), docs)
    if missing:
        report.points_non_verifiables = missing + report.points_non_verifiables
        report.rapport_texte = render_text(report)
    return report


def submission_verdict(db: Session, citizen: Citizen, application: Application) -> SubmissionVerdict:
    """The verdict frozen onto a submitted dossier — rules only, no LLM.

    Rules alone because this verdict leaves the citizen's hands: it must be
    reproducible (same dossier, same verdict) and instant at submission, and
    the local agent never removes a discrepancy anyway — it only comments on it.
    """
    docs = [d for d in (_ensure_analysis(db, doc) for doc in application.documents) if d]
    engine = run_rules(declared_profile(citizen), docs)
    by_severity = {g: sum(1 for i in engine.inconsistencies if i.gravite == g)
                   for g in ("elevee", "moyenne", "faible")}
    readable = [d for d in docs if d.text.strip()]
    if not readable:
        statut = "non_verifiable"
    elif engine.inconsistencies:
        statut = "ecarts_non_resolus"
    else:
        statut = "coherent"
    return SubmissionVerdict(
        statut=statut,
        nb_ecarts=len(engine.inconsistencies),
        nb_elevee=by_severity["elevee"],
        nb_moyenne=by_severity["moyenne"],
        nb_faible=by_severity["faible"],
        nb_documents=len(docs),
        nb_points_non_verifiables=len(engine.unverifiable),
        calcule_le=datetime.now(UTC),
    )

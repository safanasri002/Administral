"""HTTP layer for local inconsistency detection."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.ai.inconsistency.schemas import InconsistencyReport
from app.modules.ai.inconsistency.service import build_report
from app.modules.auth.dependencies import require_citizen
from app.modules.auth.models import User
from app.modules.citizen.profile import resolve_citizen

router = APIRouter(tags=["ai-inconsistency"])


@router.get(
    "/citizen/incoherences",
    response_model=InconsistencyReport,
    response_model_by_alias=True,
    summary="Incohérences entre mon profil et mes documents",
    description=(
        "Compare le profil déclaré du citoyen connecté aux informations extraites "
        "de ses documents (métadonnées, texte natif ou OCR local, règles), puis "
        "soumet les cas ambigus à un modèle Mistral local. Tout le traitement a "
        "lieu sur le serveur : aucun document n'est envoyé à un service tiers. "
        "Chaque écart est « à vérifier » et cite l'extrait qui le fonde."
    ),
)
def get_inconsistency_report(
    current_user: User = Depends(require_citizen),
    db: Session = Depends(get_db),
) -> InconsistencyReport:
    citizen = resolve_citizen(db, current_user)
    return build_report(db, citizen)

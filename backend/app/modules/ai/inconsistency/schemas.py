"""Structured data for local inconsistency detection.

Two shapes matter:

  * `DocumentFacts` — what was read from one document, every value carrying the
    verbatim excerpt it came from. Persisted as JSONB on the document at upload
    so a report never re-runs OCR.
  * `InconsistencyReport` — the comparison result, one `Inconsistency` per
    discrepancy, each "à vérifier" with its evidence.

Wire format is camelCase like the rest of the API; the domain vocabulary stays
French.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

#: Bumped when extraction rules change, so stale stored facts can be recomputed.
FACTS_VERSION = "inc-v1"

FactSource = Literal["regle", "metadonnees", "agent_ia"]
TextMethod = Literal["texte_natif", "ocr_local", "mixte", "fourni", "aucune"]
Scope = Literal["profil_document", "document_document"]
Severity = Literal["elevee", "moyenne", "faible"]
AgentOpinion = Literal["confirme", "infirme", "incertain"]


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class Fact(CamelModel):
    """One extracted value and the excerpt that proves it."""

    valeur: str
    #: Verbatim line from the document. Shown to the citizen as the evidence.
    preuve: str | None = None
    source: FactSource = "regle"
    confiance: float = Field(default=1.0, ge=0.0, le=1.0)


class GpsPosition(CamelModel):
    latitude: float
    longitude: float


class DocumentMetadata(CamelModel):
    format: str | None = None
    auteur: str | None = None
    titre: str | None = None
    sujet: str | None = None
    logiciel: str | None = None
    producteur: str | None = None
    date_creation: str | None = None
    date_modification: str | None = None
    nb_pages: int | None = None
    appareil: str | None = None
    date_prise_vue: str | None = None
    gps: GpsPosition | None = None
    #: Which reader produced these values: `pymupdf`, `pillow`, `exiftool`.
    extracteur: str | None = None


class DocumentFacts(CamelModel):
    version: str = FACTS_VERSION
    fichier: str
    methode_texte: TextMethod = "aucune"
    pages_ocr: list[int] = Field(default_factory=list)
    nb_caracteres: int = 0
    metadonnees: DocumentMetadata = Field(default_factory=DocumentMetadata)

    type_document: Fact | None = None
    statut_occupation: Fact | None = None
    noms: list[Fact] = Field(default_factory=list)
    adresses: list[Fact] = Field(default_factory=list)
    codes_postaux: list[Fact] = Field(default_factory=list)
    dates: list[Fact] = Field(default_factory=list)
    loyer: Fact | None = None
    surface: Fact | None = None

    avertissements: list[str] = Field(default_factory=list)


class DocumentValue(CamelModel):
    fichier: str
    valeur: str | None = None
    preuve: str | None = None


class Inconsistency(CamelModel):
    #: Stable within one report; the agent refers to findings by it.
    id: str
    portee: Scope
    champ: str
    gravite: Severity
    valeur_profil: str | None = None
    valeurs_documents: list[DocumentValue] = Field(default_factory=list)
    message: str
    #: Always "a_verifier": the engine flags, the citizen (or an agent) decides.
    statut: Literal["a_verifier"] = "a_verifier"
    source: FactSource = "regle"
    avis_agent: AgentOpinion | None = None
    explication_agent: str | None = None


class AgentStatus(CamelModel):
    active: bool
    disponible: bool = False
    modele: str | None = None
    utilise: bool = False
    note: str | None = None


class InconsistencyReport(CamelModel):
    dossier_id: str
    genere_le: datetime
    nb_documents_analyses: int
    incoherences: list[Inconsistency]
    #: What could not be checked and why — absence of a finding is not proof.
    points_non_verifiables: list[str] = Field(default_factory=list)
    agent_ia: AgentStatus
    #: Plain-text rendering of the report, ready to display or export.
    rapport_texte: str


class SubmissionVerdict(CamelModel):
    """What the administration may learn about coherence: the verdict on the
    dossier *as submitted*, nothing more.

    Frozen once, at submission. Deliberately minimal — status and counts only,
    no discrepancy, no evidence, no trace of the checks the citizen ran before
    sending (data minimisation: the citizen's drafts are theirs). Computed by
    the deterministic rules alone, so it is reproducible and explainable.
    """

    #: `non_verifiable` when no submitted document had readable text.
    statut: Literal["coherent", "ecarts_non_resolus", "non_verifiable"]
    nb_ecarts: int = 0
    nb_elevee: int = 0
    nb_moyenne: int = 0
    nb_faible: int = 0
    nb_documents: int = 0
    nb_points_non_verifiables: int = 0
    methode: Literal["regles"] = "regles"
    version: str = FACTS_VERSION
    calcule_le: datetime

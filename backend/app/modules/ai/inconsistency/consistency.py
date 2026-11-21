"""Consistency engine: profile ↔ document and document ↔ document rules.

Every rule is explicit and explainable — the finding names the declared value,
the observed one, and the line it was read from. Rules flag; they do not
conclude. A rule that cannot tell a real discrepancy from a reading artefact
(rent with or without charges, a city only partly read by OCR) marks its
finding *ambiguous*, and those go to the local Mistral agent for a second look.

Tolerances lean toward silence: a report full of false alarms trains people to
ignore it, which is worse than missing a marginal case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.modules.ai.inconsistency.information_extraction import STATUS_LABELS, fold
from app.modules.ai.inconsistency.schemas import (
    DocumentFacts,
    DocumentValue,
    Inconsistency,
    Severity,
)

#: Document types that describe the dwelling, hence its address and status.
HOUSING_TYPES = {"bail", "quittance_loyer", "attestation_hebergement", "facture", "titre_propriete"}
#: Document types that name their holder.
PERSONAL_TYPES = HOUSING_TYPES | {
    "avis_imposition", "bulletin_salaire", "piece_identite", "rib",
}
RENT_TYPES = {"bail", "quittance_loyer"}
_RENTAL_FAMILY = {"locataire", "colocataire", "sous_locataire"}
#: Below this, a text is too short (or too badly OCR'd) to prove an absence.
_MIN_TEXT_FOR_ABSENCE = 200
_STREET_WORDS = {
    "rue", "avenue", "av", "boulevard", "bd", "chemin", "allee", "impasse", "place", "route",
    "quai", "cours", "square", "residence", "lotissement", "voie", "cite", "bis", "ter",
    "des", "les", "del", "de", "du", "la", "le",
}


@dataclass
class DeclaredProfile:
    prenom: str | None = None
    nom: str | None = None
    situation_logement: str | None = None
    type_location: str | None = None
    adresse: str | None = None
    code_postal: str | None = None
    ville: str | None = None
    loyer_mensuel: float | None = None
    surface_m2: float | None = None

    @property
    def expected_status(self) -> str | None:
        """Profile vocabulary → document vocabulary."""
        if self.situation_logement == "locataire":
            return {"colocation": "colocataire", "sous_location": "sous_locataire"}.get(
                self.type_location or "", "locataire"
            )
        return self.situation_logement  # proprietaire / heberge / None


@dataclass
class AnalysedDocument:
    facts: DocumentFacts
    text: str

    @property
    def name(self) -> str:
        return self.facts.fichier

    @property
    def doc_type(self) -> str | None:
        return self.facts.type_document.valeur if self.facts.type_document else None


@dataclass
class EngineResult:
    inconsistencies: list[Inconsistency] = field(default_factory=list)
    #: Ids of findings the rules could not settle alone.
    ambiguous: set[str] = field(default_factory=set)
    unverifiable: list[str] = field(default_factory=list)


def _status_label(status: str | None) -> str | None:
    return STATUS_LABELS.get(status or "", status)


def _status_severity(a: str, b: str) -> Severity:
    # Tenant vs co-tenant is a nuance of the same situation; tenant vs owner is not.
    return "moyenne" if {a, b} <= _RENTAL_FAMILY else "elevee"


def _euros(value: float) -> str:
    """French money formatting: `650,00 €`."""
    return f"{value:,.2f}".replace(",", " ").replace(".", ",") + " €"


def _amounts_differ(a: float, b: float, *, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) > max(abs_tol, rel_tol * max(a, b))


def _word_present(word: str, text_words: set[str], threshold: float = 0.8) -> bool:
    """Exact or near-exact presence — tolerates one OCR slip in a name of five
    letters or more (`Durand` read `Duraud` scores exactly 0.8)."""
    if word in text_words:
        return True
    return any(
        abs(len(w) - len(word)) <= 2 and SequenceMatcher(None, word, w).ratio() >= threshold
        for w in text_words
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", fold(text)))


class _Builder:
    def __init__(self) -> None:
        self.result = EngineResult()

    def add(self, *, ambiguous: bool = False, **kwargs) -> None:
        finding = Inconsistency(id=f"INC-{len(self.result.inconsistencies) + 1:02d}", **kwargs)
        self.result.inconsistencies.append(finding)
        if ambiguous:
            self.result.ambiguous.add(finding.id)


def _check_status(profile: DeclaredProfile, docs: list[AnalysedDocument], b: _Builder) -> None:
    expected = profile.expected_status
    with_status = [d for d in docs if d.facts.statut_occupation]
    for doc in docs:
        if doc.doc_type in {"bail", "attestation_hebergement", "titre_propriete"} and not doc.facts.statut_occupation:
            b.result.unverifiable.append(
                f"{doc.name} : statut d'occupation non identifié dans le document."
            )
    if expected is None:
        if with_status:
            b.result.unverifiable.append(
                "Statut d'occupation non déclaré dans le profil : comparaison impossible."
            )
    else:
        for doc in with_status:
            fact = doc.facts.statut_occupation
            assert fact is not None
            if fact.valeur != expected:
                b.add(
                    portee="profil_document",
                    champ="statut_occupation",
                    gravite=_status_severity(expected, fact.valeur),
                    valeur_profil=_status_label(expected),
                    valeurs_documents=[
                        DocumentValue(fichier=doc.name, valeur=_status_label(fact.valeur), preuve=fact.preuve)
                    ],
                    message=(
                        f"Profil : {_status_label(expected)} — Document : "
                        f"{_status_label(fact.valeur)} ({doc.name})."
                    ),
                    ambiguous=fact.confiance < 0.8 or {expected, fact.valeur} <= _RENTAL_FAMILY,
                )

    # Document ↔ document: two pieces describing different occupancy.
    distinct = {d.facts.statut_occupation.valeur for d in with_status}  # type: ignore[union-attr]
    if len(distinct) > 1:
        values = [
            DocumentValue(
                fichier=d.name,
                valeur=_status_label(d.facts.statut_occupation.valeur),  # type: ignore[union-attr]
                preuve=d.facts.statut_occupation.preuve,  # type: ignore[union-attr]
            )
            for d in with_status
        ]
        labels = sorted(_status_label(s) or s for s in distinct)
        b.add(
            portee="document_document",
            champ="statut_occupation",
            gravite="moyenne" if distinct <= _RENTAL_FAMILY else "elevee",
            valeurs_documents=values,
            message=f"Les documents décrivent des statuts différents : {', '.join(labels)}.",
            ambiguous=distinct <= _RENTAL_FAMILY,
        )


def _check_address(profile: DeclaredProfile, docs: list[AnalysedDocument], b: _Builder) -> None:
    housing = [d for d in docs if d.doc_type in HOUSING_TYPES]
    for doc in housing:
        found_codes = {f.valeur.split()[0]: f for f in doc.facts.codes_postaux}
        if not found_codes:
            if len(doc.text) >= _MIN_TEXT_FOR_ABSENCE:
                b.result.unverifiable.append(f"{doc.name} : aucune adresse postale identifiée.")
            continue
        observed = [
            DocumentValue(fichier=doc.name, valeur=f.valeur, preuve=f.preuve)
            for f in found_codes.values()
        ]
        if profile.code_postal and profile.code_postal not in found_codes:
            b.add(
                portee="profil_document",
                champ="adresse",
                gravite="elevee",
                valeur_profil=" ".join(x for x in (profile.code_postal, profile.ville) if x),
                valeurs_documents=observed,
                message=(
                    f"Le code postal déclaré ({profile.code_postal}) n'apparaît dans aucune "
                    f"adresse de {doc.name}."
                ),
                # Several addresses on one piece (landlord, agency) need reading.
                ambiguous=len(found_codes) > 1,
            )
            continue
        if not profile.code_postal and profile.ville and fold(profile.ville) not in fold(doc.text):
            b.add(
                portee="profil_document",
                champ="adresse",
                gravite="moyenne",
                valeur_profil=profile.ville,
                valeurs_documents=observed,
                message=f"La ville déclarée ({profile.ville}) n'est pas mentionnée dans {doc.name}.",
                ambiguous=True,
            )
            continue
        # Same postcode: is it the same street?
        street_words = _words(profile.adresse or "") - _STREET_WORDS
        street_words = {w for w in street_words if len(w) >= 3 and not w.isdigit()}
        if street_words and doc.facts.adresses:
            text_words = _words(doc.text)
            missing = [w for w in street_words if not _word_present(w, text_words)]
            if len(missing) > len(street_words) / 2:
                b.add(
                    portee="profil_document",
                    champ="adresse",
                    gravite="faible",
                    valeur_profil=profile.adresse,
                    valeurs_documents=[
                        DocumentValue(fichier=doc.name, valeur=f.valeur, preuve=f.preuve)
                        for f in doc.facts.adresses
                    ],
                    message=(
                        f"Même code postal, mais la rue déclarée ne semble pas figurer dans {doc.name}."
                    ),
                    ambiguous=True,
                )

    # Document ↔ document: housing pieces with no address in common.
    coded = [(d, {f.valeur.split()[0] for f in d.facts.codes_postaux}) for d in housing]
    coded = [(d, codes) for d, codes in coded if codes]
    for i, (first, codes_a) in enumerate(coded):
        for second, codes_b in coded[i + 1 :]:
            if codes_a.isdisjoint(codes_b):
                b.add(
                    portee="document_document",
                    champ="adresse",
                    gravite="moyenne",
                    valeurs_documents=[
                        DocumentValue(fichier=d.name, valeur=f.valeur, preuve=f.preuve)
                        for d in (first, second)
                        for f in d.facts.codes_postaux
                    ],
                    message=f"{first.name} et {second.name} ne partagent aucun code postal.",
                    ambiguous=True,
                )


def _check_name(profile: DeclaredProfile, docs: list[AnalysedDocument], b: _Builder) -> None:
    if not profile.nom:
        return
    name_words = {w for w in _words(profile.nom) if len(w) >= 2}
    if not name_words:
        return
    full_name = " ".join(x for x in (profile.prenom, profile.nom) if x)
    for doc in docs:
        if doc.doc_type not in PERSONAL_TYPES:
            continue
        if len(doc.text) < _MIN_TEXT_FOR_ABSENCE:
            b.result.unverifiable.append(
                f"{doc.name} : texte trop court pour vérifier l'identité du titulaire."
            )
            continue
        text_words = _words(doc.text)
        if all(_word_present(w, text_words) for w in name_words):
            continue
        b.add(
            portee="profil_document",
            champ="identite",
            gravite="moyenne",
            valeur_profil=full_name,
            valeurs_documents=[
                DocumentValue(fichier=doc.name, valeur=f.valeur, preuve=f.preuve)
                for f in doc.facts.noms
            ]
            or [DocumentValue(fichier=doc.name, valeur=None)],
            message=f"Le nom déclaré ({full_name}) n'a pas été retrouvé dans {doc.name}.",
            # OCR may have mangled the name: worth a second reading.
            ambiguous=doc.facts.methode_texte in {"ocr_local", "mixte"} or bool(doc.facts.noms),
        )


def _check_rent(profile: DeclaredProfile, docs: list[AnalysedDocument], b: _Builder) -> None:
    rents = [(d, d.facts.loyer) for d in docs if d.doc_type in RENT_TYPES and d.facts.loyer]
    if profile.loyer_mensuel:
        for doc, fact in rents:
            observed = float(fact.valeur)  # type: ignore[union-attr]
            if _amounts_differ(profile.loyer_mensuel, observed, abs_tol=10, rel_tol=0.05):
                b.add(
                    portee="profil_document",
                    champ="loyer",
                    gravite="moyenne",
                    valeur_profil=_euros(profile.loyer_mensuel),
                    valeurs_documents=[
                        DocumentValue(fichier=doc.name, valeur=_euros(observed), preuve=fact.preuve)  # type: ignore[union-attr]
                    ],
                    message=(
                        f"Loyer déclaré {_euros(profile.loyer_mensuel)} — {doc.name} indique "
                        f"{_euros(observed)}."
                    ),
                    # "charges comprises" vs "hors charges" is the classic false alarm.
                    ambiguous="charge" in fold(fact.preuve or ""),  # type: ignore[union-attr]
                )
    amounts = {float(f.valeur) for _, f in rents}  # type: ignore[union-attr]
    if len(rents) > 1 and max(amounts) - min(amounts) > max(10, 0.05 * max(amounts)):
        b.add(
            portee="document_document",
            champ="loyer",
            gravite="faible",
            valeurs_documents=[
                DocumentValue(fichier=d.name, valeur=_euros(float(f.valeur)), preuve=f.preuve)  # type: ignore[union-attr]
                for d, f in rents
            ],
            message="Les montants de loyer diffèrent d'un document à l'autre.",
            # A yearly rent revision is legitimate; the dates tell.
            ambiguous=True,
        )


def _check_surface(profile: DeclaredProfile, docs: list[AnalysedDocument], b: _Builder) -> None:
    if not profile.surface_m2:
        return
    for doc in docs:
        fact = doc.facts.surface
        if doc.doc_type != "bail" or fact is None:
            continue
        observed = float(fact.valeur)
        if _amounts_differ(profile.surface_m2, observed, abs_tol=2, rel_tol=0.10):
            b.add(
                portee="profil_document",
                champ="surface",
                gravite="faible",
                valeur_profil=f"{profile.surface_m2:g} m²",
                valeurs_documents=[
                    DocumentValue(fichier=doc.name, valeur=f"{observed:g} m²", preuve=fact.preuve)
                ],
                message=f"Surface déclarée {profile.surface_m2:g} m² — {doc.name} indique {observed:g} m².",
            )


def run_rules(profile: DeclaredProfile, docs: list[AnalysedDocument]) -> EngineResult:
    builder = _Builder()
    for doc in docs:
        if not doc.text.strip():
            builder.result.unverifiable.append(
                f"{doc.name} : aucun texte exploitable (scan sans OCR local, ou document vide)."
            )
    readable = [d for d in docs if d.text.strip()]
    _check_status(profile, readable, builder)
    _check_address(profile, readable, builder)
    _check_name(profile, readable, builder)
    _check_rent(profile, readable, builder)
    _check_surface(profile, readable, builder)
    return builder.result

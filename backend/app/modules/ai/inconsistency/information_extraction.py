"""Rule-based information extraction from document text.

Deterministic on purpose: the same document always yields the same facts, and
each fact carries the verbatim line it was read from — that line is what the
report shows as "Preuve détectée". The local LLM is only asked about cases the
rules leave ambiguous (see `local_agent`).

Matching happens on a *folded* copy of the text (lowercase, accents stripped)
that keeps one character per original character, so a match position in the
folded text is also a position in the original — evidence is quoted as written.
"""

from __future__ import annotations

import re
import unicodedata

from app.modules.ai.inconsistency.schemas import DocumentFacts, DocumentMetadata, Fact

_MAX_FACTS = 5


def _fold_char(ch: str) -> str:
    base = [c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)]
    folded = (base[0] if base else ch).lower()
    return folded if len(folded) == 1 else ch


def fold(text: str) -> str:
    """Lowercase, accent-free, *same length* as the input."""
    return "".join(_fold_char(ch) for ch in text)


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    line = text[start : end if end != -1 else len(text)].strip()
    return re.sub(r"\s+", " ", line)[:200]


# --- Document type ---------------------------------------------------------
# The type whose marker appears *earliest* wins: a document's title comes
# first, while other types' words show up later in clauses (a lease mentions
# "quittance", a rent receipt rarely says "contrat de location").
_DOC_TYPES: dict[str, str] = {
    "bail": r"contrat de (?:location|bail)|bail (?:d.habitation|de colocation|meuble)",
    "quittance_loyer": r"quittance de loyer|\bquittance\b",
    "attestation_hebergement": r"attestation d.?hebergement|certifie (?:sur l.honneur )?heberger",
    "avis_imposition": r"avis d.?impot|impot sur (?:le|les) revenus?|revenu fiscal de reference",
    "bulletin_salaire": r"bulletin de (?:paie|salaire)|net a payer",
    "titre_propriete": r"acte (?:authentique )?de vente|taxe fonciere|titre de propriete",
    "piece_identite": r"carte nationale d.?identite|\bpasseport\b|titre de sejour",
    "facture": r"\bfacture\b",
    "rib": r"releve d.?identite bancaire|\biban\b",
}

# --- Occupancy status ------------------------------------------------------
# Priority order matters: "colocataire" must win over "locataire", and the
# lookbehind keeps "colocataire"/"sous-locataire" from matching plain
# "locataire". Lines that *forbid* something are skipped — nearly every French
# lease says "la sous-location est interdite", which says nothing about the
# tenant's own status.
# Within one status, patterns are tried in order: the role noun naming the
# person ("Mme X, colocataire") is better evidence than the activity noun in a
# title ("bail de colocation"), so it is quoted when both are present.
_STATUSES: list[tuple[str, tuple[str, ...]]] = [
    ("colocataire", (r"\bco-?locataires?\b", r"\bcolocation\b|bail solidaire")),
    ("sous_locataire", (r"\bsous[- ]locataires?\b", r"\bsous[- ]location\b")),
    ("heberge", (r"\bheberge(?:e|s|es)?\b", r"attestation d.?hebergement|certifie (?:sur l.honneur )?heberger")),
    ("proprietaire", (r"proprietaire occupant|je suis proprietaire", r"taxe fonciere|acte (?:authentique )?de vente")),
    ("locataire", (r"(?<![\w-])locataires?\b", r"\bpreneurs?\b")),
]
_PROHIBITION = re.compile(
    r"interdit|interdiction|ne (?:peut|pourra|saurait)|sans (?:l.)?accord|autoris|en cas de"
)

STATUS_LABELS = {
    "locataire": "Locataire",
    "colocataire": "Colocataire",
    "sous_locataire": "Sous-locataire",
    "heberge": "Hébergé(e)",
    "proprietaire": "Propriétaire",
}

# --- Identity, address, amounts, dates -------------------------------------
_NAME = re.compile(
    r"(?:\bM\.|\bMr\.?|\bMme\.?|\bMlle\.?|\bMadame|\bMonsieur|Nom(?: et pr[ée]nom)?\s*:"
    r"|Locataire\s*:|Preneur\s*:|B[ée]n[ée]ficiaire\s*:)\s*"
    r"([A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+(?:[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+){0,3})"
)
_STREET = re.compile(
    r"\b\d{1,4}\s*(?:bis|ter)?,?\s+(?:rue|avenue|av\.?|boulevard|bd|chemin|all[ée]e|impasse"
    r"|place|route|quai|cours|square|r[ée]sidence|lotissement|voie|cit[ée])\b[^\n]{2,80}",
    re.IGNORECASE,
)
# French postal codes (01–98), not glued to a longer number, followed by a town
# — and not by a currency, which is what "15000 euros" would otherwise give.
_POSTCODE = re.compile(
    r"(?<![\d,.])((?:0[1-9]|[1-8]\d|9[0-8])\d{3})(?!\d)[ \t]+"
    r"(?!(?:euros?|eur|€)\b)([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' \-]{1,40})",
    re.IGNORECASE,
)
_AMOUNT = r"(\d{1,3}(?:[ .  ]\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur\b|euros?\b)"
_SURFACE = re.compile(r"(\d{1,4}(?:[,.]\d{1,2})?)\s*m(?:²|2)(?!\w)", re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")
_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}
_TEXT_DATE = re.compile(r"\b(\d{1,2})(?:er)?\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b")


def parse_amount(raw: str) -> float | None:
    """`1 250,50` / `1.250,50` / `650.00` / `650` → float."""
    s = re.sub(r"[   ]", "", raw)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.search(r"\.\d{3}$", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _doc_type(text: str, folded: str) -> Fact | None:
    best: tuple[int, str] | None = None
    for doc_type, pattern in _DOC_TYPES.items():
        match = re.search(pattern, folded)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), doc_type)
    if best is None:
        return None
    return Fact(valeur=best[1], preuve=_line_at(text, best[0]), confiance=0.8)


def _status(text: str, folded: str) -> Fact | None:
    for status, patterns in _STATUSES:
        for pattern in patterns:
            for match in re.finditer(pattern, folded):
                if _PROHIBITION.search(_line_at(folded, match.start())):
                    continue
                return Fact(valeur=status, preuve=_line_at(text, match.start()), confiance=0.85)
    return None


def _names(text: str) -> list[Fact]:
    facts: dict[str, Fact] = {}
    for match in _NAME.finditer(text):
        name = match.group(1).strip()
        facts.setdefault(fold(name), Fact(valeur=name, preuve=_line_at(text, match.start())))
    return list(facts.values())[:_MAX_FACTS]


def _addresses(text: str) -> tuple[list[Fact], list[Fact]]:
    streets = [
        Fact(valeur=re.sub(r"\s+", " ", m.group(0)).strip(" ,;"), preuve=_line_at(text, m.start()))
        for m in _STREET.finditer(text)
    ]
    postcodes: dict[str, Fact] = {}
    for m in _POSTCODE.finditer(text):
        town = re.split(r"\s{2,}|\t", m.group(2).strip())[0].strip(" -'")
        postcodes.setdefault(
            m.group(1), Fact(valeur=f"{m.group(1)} {town}".strip(), preuve=_line_at(text, m.start()))
        )
    return streets[:_MAX_FACTS], list(postcodes.values())[:_MAX_FACTS]


def _rent(text: str, folded: str) -> Fact | None:
    """The rent amount — preferring the line that names it most precisely."""
    candidates: list[tuple[int, Fact]] = []
    for m in re.finditer(r"\bloyer\b[^\n]*", folded):
        line_folded = m.group(0)
        amount = re.search(_AMOUNT, line_folded)
        if not amount:
            continue
        value = parse_amount(amount.group(1))
        if value is None or value <= 0:
            continue
        rank = 0 if "hors charges" in line_folded else 1 if "mensuel" in line_folded else 2
        candidates.append(
            (rank, Fact(valeur=f"{value:.2f}", preuve=_line_at(text, m.start()), confiance=0.8))
        )
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def _surface(text: str, folded: str) -> Fact | None:
    matches = list(_SURFACE.finditer(text))
    if not matches:
        return None
    preferred = [
        m for m in matches if re.search(r"surface|superficie|habitable", _line_at(folded, m.start()))
    ]
    m = (preferred or matches)[0]
    value = parse_amount(m.group(1))
    if value is None:
        return None
    return Fact(valeur=f"{value:g}", preuve=_line_at(text, m.start()), confiance=0.75)


def _dates(text: str, folded: str) -> list[Fact]:
    found: dict[str, Fact] = {}
    for m in _NUMERIC_DATE.finditer(text):
        d, mo, y = (int(x) for x in m.groups())
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100:
            found.setdefault(f"{y:04d}-{mo:02d}-{d:02d}", Fact(valeur=f"{y:04d}-{mo:02d}-{d:02d}", preuve=_line_at(text, m.start())))
    for m in _TEXT_DATE.finditer(folded):
        d, mo, y = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
        if 1 <= d <= 31:
            found.setdefault(f"{y:04d}-{mo:02d}-{d:02d}", Fact(valeur=f"{y:04d}-{mo:02d}-{d:02d}", preuve=_line_at(text, m.start())))
    return sorted(found.values(), key=lambda f: f.valeur)[:10]


def extract_facts(
    text: str,
    file_name: str,
    *,
    method: str = "aucune",
    ocr_pages: list[int] | None = None,
    metadata: DocumentMetadata | None = None,
    warnings: list[str] | None = None,
) -> DocumentFacts:
    """Structure one document's text into comparable facts."""
    folded = fold(text)
    streets, postcodes = _addresses(text)
    facts = DocumentFacts(
        fichier=file_name,
        methode_texte=method,  # type: ignore[arg-type]
        pages_ocr=ocr_pages or [],
        nb_caracteres=len(text),
        metadonnees=metadata or DocumentMetadata(),
        type_document=_doc_type(text, folded),
        statut_occupation=_status(text, folded),
        noms=_names(text),
        adresses=streets,
        codes_postaux=postcodes,
        dates=_dates(text, folded),
        loyer=_rent(text, folded),
        surface=_surface(text, folded),
        avertissements=list(warnings or []),
    )
    # A document's own title tells its status when the body does not repeat it.
    if facts.statut_occupation is None and facts.type_document is not None:
        implied = {
            "titre_propriete": "proprietaire",
            "attestation_hebergement": "heberge",
            "quittance_loyer": "locataire",
        }.get(facts.type_document.valeur)
        if implied:
            facts.statut_occupation = Fact(
                valeur=implied, preuve=facts.type_document.preuve, confiance=0.6
            )
    return facts

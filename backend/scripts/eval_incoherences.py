"""Evaluate inconsistency detection on a labelled synthetic set.

Usage (from backend/):
    python -m scripts.eval_incoherences            # native text only
    python -m scripts.eval_incoherences --ocr      # + same documents as images, read by local OCR

Each case is one document compared with a fixed declared profile. The ground
truth lists the fields that SHOULD be flagged (profile <-> document scope).
Traps — wording that a naive rule would flag but that is not an inconsistency —
are included on purpose: a forbidden-sublet clause, a landlord living elsewhere,
a rent quoted charges included, a name misread by OCR, a 5-digit amount.

Caveat: the documents are generated from templates written by the same person
who wrote the rules. The scores measure the rules on these phrasings, not on
real-world documents — a labelled set of real (anonymised) documents is the next
step before quoting a production accuracy.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from app.modules.ai.inconsistency.consistency import AnalysedDocument, DeclaredProfile, run_rules
from app.modules.ai.inconsistency.information_extraction import extract_facts
from app.modules.ai.inconsistency.text_extraction import extract_local_text

FIELDS = ["statut_occupation", "adresse", "identite", "loyer", "surface"]

PROFILE = DeclaredProfile(
    prenom="Camille", nom="Durand", situation_logement="locataire", type_location="vide",
    adresse="12 rue des Lilas", code_postal="69003", ville="Lyon", loyer_mensuel=650.0, surface_m2=42.0,
)

FILLER = (
    "Le présent document est établi pour servir et valoir ce que de droit. "
    "Les informations ci-dessus sont certifiées exactes par le signataire. "
    "Toute fausse déclaration expose son auteur à des poursuites. "
)


@dataclass
class Doc:
    nom: str = "Camille Durand"
    rue: str = "12 rue des Lilas"
    cp: str = "69003"
    ville: str = "Lyon"
    statut: str = "locataire"
    loyer: str = "650,00"
    surface: str = "42"
    charges_comprises: bool = False


def bail(d: Doc, v: int) -> str:
    preneur = {
        "locataire": "locataire",
        "colocataire": "colocataire",
        "sous_locataire": "sous-locataire",
    }[d.statut]
    titre = ["CONTRAT DE LOCATION", "CONTRAT DE LOCATION - LOGEMENT VIDE", "BAIL D'HABITATION"][v % 3]
    if d.statut == "colocataire":
        titre = "CONTRAT DE LOCATION - BAIL DE COLOCATION"
    return (
        f"{titre}\n"
        f"Entre M. Jean Martin, bailleur, demeurant 3 rue de Rivoli 75001 Paris\n"
        f"Et Mme {d.nom}, {preneur}, ci-après le preneur.\n"
        f"Désignation : logement situé {d.rue} {d.cp} {d.ville}\n"
        f"Surface habitable : {d.surface} m2\n"
        f"Loyer mensuel hors charges : {d.loyer} EUR\n"
        f"La sous-location est interdite sans l'accord écrit du bailleur.\n"
        f"Fait à {d.ville}, le 15 janvier 2025.\n{FILLER}"
    )


def quittance(d: Doc, v: int) -> str:
    ligne_loyer = (
        f"Loyer charges comprises : {d.loyer} EUR" if d.charges_comprises else f"Loyer : {d.loyer} EUR"
    )
    return (
        f"QUITTANCE DE LOYER\n"
        f"Je soussigné M. Jean Martin, bailleur, déclare avoir reçu de Mme {d.nom}, locataire,\n"
        f"la somme due au titre du logement situé {d.rue} {d.cp} {d.ville}.\n"
        f"{ligne_loyer}\n"
        f"Période : {['mars', 'avril', 'mai'][v % 3]} 2025\n{FILLER}"
    )


def facture(d: Doc, v: int) -> str:
    return (
        f"FACTURE D'ÉLECTRICITÉ - {['EDF', 'Engie', 'TotalEnergies'][v % 3]}\n"
        f"Titulaire du contrat : Mme {d.nom}\n"
        f"Adresse de consommation : {d.rue} {d.cp} {d.ville}\n"
        f"Montant total à payer : 84,30 EUR\nDate d'émission : 02/06/2025\n{FILLER}"
    )


def avis(d: Doc, v: int) -> str:
    return (
        f"AVIS D'IMPÔT SUR LES REVENUS 2024\n"
        f"Déclarant : Mme {d.nom}\n"
        f"Adresse : {d.rue} {d.cp} {d.ville}\n"
        f"Revenu fiscal de référence : {[15000, 16800, 14250][v % 3]} euros\n{FILLER}"
    )


def attestation(d: Doc, v: int) -> str:
    return (
        f"ATTESTATION D'HÉBERGEMENT\n"
        f"Je soussigné M. Karim Haddad certifie sur l'honneur héberger à titre gratuit\n"
        f"Mme {d.nom} à mon domicile situé {d.rue} {d.cp} {d.ville}.\n{FILLER}"
    )


@dataclass
class Case:
    name: str
    text: str
    expected: set[str] = field(default_factory=set)


def build_cases() -> list[Case]:
    rng = random.Random(42)
    cases: list[Case] = []

    def add(kind, template, doc, expected, v):
        cases.append(Case(f"{template.__name__}/{kind}/v{v}", template(doc, v), set(expected)))

    for v in range(3):
        # --- Lease: every field applies -------------------------------------
        add("ok", bail, Doc(), [], v)
        add("colocataire", bail, Doc(statut="colocataire"), ["statut_occupation"], v)
        add("sous_locataire", bail, Doc(statut="sous_locataire"), ["statut_occupation"], v)
        add("autre_cp", bail, Doc(cp=rng.choice(["69006", "69007", "38000"])), ["adresse"], v)
        add("autre_rue", bail, Doc(rue="8 avenue Foch"), ["adresse"], v)
        add("autre_nom", bail, Doc(nom="Karim Ben Salah"), ["identite"], v)
        add("nom_ocr", bail, Doc(nom="Camille Duraud"), [], v)
        add("loyer_eleve", bail, Doc(loyer="720,00"), ["loyer"], v)
        add("loyer_proche", bail, Doc(loyer="655,00"), [], v)
        add("surface", bail, Doc(surface="55"), ["surface"], v)
        add("surface_proche", bail, Doc(surface="43"), [], v)
        add("cp+loyer", bail, Doc(cp="69006", loyer="720,00"), ["adresse", "loyer"], v)
        # --- Rent receipt ------------------------------------------------------
        add("ok", quittance, Doc(), [], v)
        add("autre_cp", quittance, Doc(cp="69006"), ["adresse"], v)
        add("autre_nom", quittance, Doc(nom="Karim Ben Salah"), ["identite"], v)
        add("loyer_eleve", quittance, Doc(loyer="760,00"), ["loyer"], v)
        add("charges_comprises", quittance, Doc(loyer="700,00", charges_comprises=True), [], v)
        # --- Energy bill -------------------------------------------------------
        add("ok", facture, Doc(), [], v)
        add("autre_cp", facture, Doc(cp="13001", ville="Marseille"), ["adresse"], v)
        add("autre_nom", facture, Doc(nom="Karim Ben Salah"), ["identite"], v)
        # --- Tax notice (identity only) -----------------------------------------
        add("ok", avis, Doc(), [], v)
        add("autre_nom", avis, Doc(nom="Karim Ben Salah"), ["identite"], v)
        # --- Accommodation certificate: declared tenant, document says hosted ----
        add("heberge", attestation, Doc(), ["statut_occupation"], v)
    return cases


def render_png(text: str) -> bytes:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
    lines = []
    for raw in text.splitlines():
        words, cur = raw.split(), ""
        for w in words:
            if len(cur) + len(w) > 70:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
    img = Image.new("RGB", (1600, 60 + 44 * len(lines)), "white")
    ImageDraw.Draw(img).multiline_text((40, 30), "\n".join(lines), fill="black", font=font, spacing=12)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def evaluate(cases: list[Case], ocr: bool) -> dict:
    per_field = {f: Counter() for f in FIELDS}
    exact = 0
    errors = []
    for case in cases:
        if ocr:
            local = extract_local_text(render_png(case.text), "image/png")
            text, method = local.text, "ocr_local"
        else:
            text, method = case.text, "texte_natif"
        doc = AnalysedDocument(facts=extract_facts(text, case.name, method=method), text=text)
        result = run_rules(PROFILE, [doc])
        predicted = {i.champ for i in result.inconsistencies if i.portee == "profil_document"}
        for f in FIELDS:
            truth, pred = f in case.expected, f in predicted
            per_field[f]["tp" if truth and pred else "fp" if pred else "fn" if truth else "tn"] += 1
        if predicted == case.expected:
            exact += 1
        else:
            errors.append((case.name, sorted(case.expected), sorted(predicted)))
    total = Counter()
    for c in per_field.values():
        total.update(c)
    return {"per_field": per_field, "total": total, "exact": exact, "n": len(cases), "errors": errors}


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def report(title: str, r: dict) -> None:
    t = r["total"]
    p, rec = _ratio(t["tp"], t["tp"] + t["fp"]), _ratio(t["tp"], t["tp"] + t["fn"])
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    acc = _ratio(t["tp"] + t["tn"], sum(t.values()))
    print(f"\n=== {title} — {r['n']} documents ===")
    print(f"{'champ':<20}{'VP':>4}{'FP':>4}{'FN':>4}{'VN':>5}{'précision':>11}{'rappel':>8}")
    for f, c in r["per_field"].items():
        print(f"{f:<20}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>5}"
              f"{_ratio(c['tp'], c['tp'] + c['fp']):>11.2f}{_ratio(c['tp'], c['tp'] + c['fn']):>8.2f}")
    print(f"Précision {p:.1%} · Rappel {rec:.1%} · F1 {f1:.1%} · "
          f"Exactitude par décision {acc:.1%} · Documents entièrement justes {r['exact']}/{r['n']} "
          f"({_ratio(r['exact'], r['n']):.1%})")
    for name, exp, got in r["errors"]:
        print(f"  ✗ {name}: attendu {exp} — obtenu {got}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true", help="also evaluate on images read by local OCR")
    args = parser.parse_args()
    cases = build_cases()
    report("Texte natif (PDF numérique)", evaluate(cases, ocr=False))
    if args.ocr:
        report("Images lues par OCR local (Tesseract)", evaluate(cases, ocr=True))


if __name__ == "__main__":
    sys.exit(main())

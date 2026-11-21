"""Local inconsistency detection between a citizen's profile and documents.

Everything here runs on the server — no document byte or extracted value is
sent to a third-party API:

    upload → metadata (PyMuPDF / Pillow / ExifTool)
           → text (PyMuPDF text layer, local OCR only when a page has none)
           → information extraction (rules)            → DocumentFacts (JSONB)
    report → consistency rules (profile ↔ doc, doc ↔ doc)
           → local Mistral agent for the ambiguous cases
           → InconsistencyReport

Scope is *inconsistency*, not fraud: findings are "à vérifier par l'usager",
never accusations. Document forensics live in `ai.fraud`.
"""

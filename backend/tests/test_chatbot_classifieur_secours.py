"""Repli du classifieur d'intention quand le LLM est injoignable.

Une clé Mistral bloquée (quota à 0) envoyait toutes les questions en hors-sujet,
y compris celles dont la branche n'a pas besoin du modèle (documents, montant).
"""

from __future__ import annotations

import pytest

from app.modules.chatbot.rag import orchestrator as o


@pytest.mark.parametrize(
    ("message", "attendu"),
    [
        ("bonjour comment faire procédure demande APL", "rag_general"),
        ("que sont les documents et le montant d'une demande d'APL", "documents_necessaires"),
        ("Quelles pièces justificatives fournir ?", "documents_necessaires"),
        ("combien je pourrais toucher ?", "estimation"),
        ("quels papiers pour la bourse CROUS", "rag_general"),
        ("j'ai reçu un refus, est-ce légal ?", "fondement_juridique"),
        ("quel temps fait-il demain ?", "fallback"),
    ],
)
def test_intention_par_mots_cles(message: str, attendu: str) -> None:
    assert o.intention_par_mots_cles(message) == attendu


def test_llm_injoignable_bascule_sur_les_mots_cles(monkeypatch) -> None:
    def quota_epuise(**_kwargs):
        raise RuntimeError("Status 429. Rate limit exceeded")

    monkeypatch.setattr(o, "call_llm", quota_epuise)
    etat = {"message": "quels documents pour ma demande d'APL ?", "conversation_history": []}
    assert o.route_intent_llm(etat) == "documents_necessaires"


def test_reponse_llm_illisible_reste_fallback(monkeypatch) -> None:
    """Le repli par mots-clés ne couvre que la panne : un modèle qui répond une
    intention inconnue reste un hors-sujet, comme avant."""
    monkeypatch.setattr(o, "call_llm", lambda **_kwargs: '{"intent": "inconnue"}')
    etat = {"message": "quels documents pour ma demande d'APL ?", "conversation_history": []}
    assert o.route_intent_llm(etat) == "fallback"

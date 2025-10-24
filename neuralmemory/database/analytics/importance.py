from __future__ import annotations

import logging


class ImportanceCalculator:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger: logging.Logger = logger

    def calculate(self, content: str, tags: list[str], action_items: list[str]) -> float:
        score: float = 0.5

        content_lower: str = content.lower()

        decision_keywords: list[str] = [
            "decided", "chose", "will implement", "selected", "determined",
            "concluded", "agreed", "committed", "finalized"
        ]
        for keyword in decision_keywords:
            if keyword in content_lower:
                score += 0.3
                break

        important_entities: list[str] = ["rahul", "claude", "neuralmemory", "pydantic"]
        entity_count: int = sum(1 for entity in important_entities if entity in content_lower)
        if entity_count >= 2:
            score += 0.2

        if action_items and len(action_items) > 0:
            score += 0.2

        word_count: int = len(content.split())
        if word_count > 100:
            score += 0.1

        final_score: float = min(1.0, max(0.0, score))
        self._logger.debug(f"Calculated importance score: {final_score:.2f}")

        return final_score

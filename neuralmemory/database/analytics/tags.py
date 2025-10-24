from __future__ import annotations

import logging


class TagSuggester:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger: logging.Logger = logger

    def suggest(self, content: str) -> list[str]:
        suggested_tags: list[str] = []
        content_lower: str = content.lower()

        tech_keywords: list[str] = [
            "refactoring", "pydantic", "architecture", "validation", "metadata",
            "vector", "database", "embedding", "search", "memory", "consolidation",
            "threading", "session", "query", "preprocessing", "importance",
            "python", "code", "guidelines", "model", "config", "api", "cli"
        ]
        for keyword in tech_keywords:
            if keyword in content_lower:
                suggested_tags.append(keyword)

        if "class" in content_lower or "def " in content_lower:
            suggested_tags.append("code")
        if "bug" in content_lower or "fix" in content_lower or "error" in content_lower:
            suggested_tags.append("bugfix")
        if "implement" in content_lower or "add" in content_lower or "create" in content_lower:
            suggested_tags.append("feature")

        unique_tags: list[str] = list(set(suggested_tags))[:10]
        self._logger.debug(f"Suggested tags: {unique_tags}")

        return unique_tags

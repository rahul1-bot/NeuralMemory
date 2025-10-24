from __future__ import annotations

import logging
from typing import Any, Callable

from neuralmemory.core.models import CodeReference


class CodeReferenceTracker:
    def __init__(
        self,
        collection: Any,
        enable_code_grounding: bool,
        validate_callback: Callable[[CodeReference], tuple[bool, str | None]],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._enable_code_grounding: bool = enable_code_grounding
        self._validate_reference: Callable[[CodeReference], tuple[bool, str | None]] = validate_callback
        self._logger: logging.Logger = logger

    def validate_memory_references(self, memory_id: str) -> tuple[bool, str | None]:
        if not self._enable_code_grounding:
            return True, None

        result = self._collection.get(
            ids=[memory_id],
            include=["metadatas"]
        )

        if not result or not result.get("ids"):
            return True, None

        metadata: dict[str, Any] = result["metadatas"][0]
        code_refs_str: str = metadata.get("code_references", "")

        if not code_refs_str:
            return True, None

        try:
            import json
            code_refs_data: list[dict[str, Any]] = json.loads(code_refs_str)
            references: list[CodeReference] = [
                CodeReference(**ref_data) for ref_data in code_refs_data
            ]

            for ref in references:
                is_valid: bool
                stale_reason: str | None
                is_valid, stale_reason = self._validate_reference(ref)

                if not is_valid:
                    self._logger.warning(
                        f"Memory {memory_id} has stale code reference: {stale_reason}"
                    )
                    return False, stale_reason

            return True, None

        except Exception as e:
            self._logger.error(f"Failed to validate code references for {memory_id}: {e}")
            return True, None

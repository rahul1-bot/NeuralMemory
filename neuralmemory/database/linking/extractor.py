from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from neuralmemory.core.models import CodeReference


class CodeReferenceExtractor:
    def __init__(
        self,
        enable_code_grounding: bool,
        logger: logging.Logger
    ) -> None:
        self._enable_code_grounding: bool = enable_code_grounding
        self._logger: logging.Logger = logger

    def extract(self, content: str) -> list[CodeReference]:
        if not self._enable_code_grounding:
            return []

        references: list[CodeReference] = []

        file_pattern: str = r'(?:^|[\s\(])([/\w]+/[\w/.]+\.py|[\w_]+\.py)'
        function_pattern: str = r'def\s+(\w+)|(\w+)\(\)'
        class_pattern: str = r'class\s+(\w+)'

        for match in re.finditer(file_pattern, content):
            file_path: str = match.group(1)
            if not file_path.startswith('/'):
                project_root: Path = Path.cwd()
                absolute_path: Path = project_root / file_path
                if absolute_path.exists():
                    file_path = str(absolute_path)

            references.append(
                CodeReference(
                    file_path=file_path,
                    last_validated=datetime.now()
                )
            )

        for match in re.finditer(function_pattern, content):
            func_name: str = match.group(1) or match.group(2)
            if func_name and len(references) > 0:
                last_ref: CodeReference = references[-1]
                references[-1] = CodeReference(
                    file_path=last_ref.file_path,
                    function_name=func_name,
                    last_validated=datetime.now()
                )

        for match in re.finditer(class_pattern, content):
            class_name: str = match.group(1)
            if class_name and len(references) > 0:
                last_ref: CodeReference = references[-1]
                references[-1] = CodeReference(
                    file_path=last_ref.file_path,
                    class_name=class_name,
                    last_validated=datetime.now()
                )

        self._logger.debug(f"Extracted {len(references)} code references from content")
        return references

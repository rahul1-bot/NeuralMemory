from __future__ import annotations

import ast
import logging
import os

from neuralmemory.core.models import CodeReference


class CodeReferenceValidator:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger: logging.Logger = logger

    def validate(self, ref: CodeReference) -> tuple[bool, str | None]:
        if not os.path.exists(ref.file_path):
            return False, f"File not found: {ref.file_path}"

        if not ref.function_name and not ref.class_name:
            return True, None

        try:
            with open(ref.file_path, 'r') as f:
                tree: ast.AST = ast.parse(f.read())

            functions: set[str] = set()
            classes: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.add(node.name)

            if ref.function_name and ref.function_name not in functions:
                return False, f"Function '{ref.function_name}' not found in {ref.file_path}"

            if ref.class_name and ref.class_name not in classes:
                return False, f"Class '{ref.class_name}' not found in {ref.file_path}"

            return True, None

        except (SyntaxError, UnicodeDecodeError) as e:
            self._logger.warning(f"Could not parse {ref.file_path}: {e}")
            return True, None

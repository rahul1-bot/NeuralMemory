from __future__ import annotations
import logging
from pathlib import Path

from neuralmemory.core.logging_setup import LoggerSetup


class MemoryTextProcessor:
    def __init__(self) -> None:
        self._logger: logging.Logger = LoggerSetup.get_logger(
            "MemoryTextProcessor",
            Path(__file__).parent / "logs" / "memory_processor.log"
        )
    
    def parse_comma_separated(self, value: str) -> list[str]:
        self._logger.debug(f"Parsing comma-separated value: {value[:50]}...")
        
        items: list[str] = []
        current: str = ""
        in_quotes: bool = False
        quote_char: str | None = None
        
        for char in value:
            if char == '"' and not in_quotes:
                in_quotes = True
                quote_char = char
                current += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                current += char
                quote_char = None
            elif char == "," and not in_quotes:
                items.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            items.append(current.strip())
        
        cleaned_items: list[str] = []
        for item in items:
            if item.startswith('"') and item.endswith('"'):
                processed_item: str = self._process_escape_sequences(item[1:-1])
                cleaned_items.append(processed_item)
            else:
                cleaned_items.append(item)
        
        self._logger.info(f"Parsed {len(cleaned_items)} items from comma-separated input")
        return cleaned_items
    
    def _process_escape_sequences(self, text: str) -> str:
        self._logger.debug("Processing escape sequences")
        
        processed: str = text
        processed = processed.replace("\\n", "\n")
        processed = processed.replace("\\t", "\t")
        processed = processed.replace("\\\\", "\\")
        processed = processed.replace('\\"', '"')
        
        return processed


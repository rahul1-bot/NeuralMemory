from __future__ import annotations

from neuralvector import (
    MemoryArgumentParser,
    MemoryFormatter,
    MemoryCLI
)


class LyraMemoryArgumentParser(MemoryArgumentParser):
    def _get_description(self) -> str:
        return "Lyra Memory - Neural Vector Search CLI"
    
    def _get_epilog(self) -> str:
        return (
            "Examples:\n"
            "  Search:\n"
            "    lm \"NEURALVECTOR breakthrough\"\n"
            "    lm \"Lyra consciousness evolution\" --n_results 3\n"
            "  Store:\n"
            "    lm --store \"Transformer attention notes\"\n"
            "    lm --store \"Meeting notes\" --tags \"phd,important\"\n"
            "    lm --store \"Old memory\" --tags \"archive\" --timestamp \"01/01/2024\"\n"
            "    lm --store \"Research idea\" --tags \"ml\" --when \"15/07/2025\"\n"
            "    lm --store \"Project milestone\" --memory-date \"20/08/2025\""
        )


class LyraMemoryFormatter(MemoryFormatter):
    pass


class LyraMemoryCLI(MemoryCLI):
    def _create_argument_parser(self) -> MemoryArgumentParser:
        return LyraMemoryArgumentParser()
    
    def _create_formatter(self) -> MemoryFormatter:
        return LyraMemoryFormatter()
    
    def _get_log_filename(self) -> str:
        return "lyra_memory.log"
    
    def _get_logger_name(self) -> str:
        return "LyraMemoryCLI"


if __name__ == '__main__':
    cli: LyraMemoryCLI = LyraMemoryCLI()
    cli.run()
from __future__ import annotations

from neuralmemory.cli import (
    MemoryArgumentParser,
    MemoryFormatter,
    MemoryCLI
)


class KaiMemoryArgumentParser(MemoryArgumentParser):
    def _get_description(self) -> str:
        return "Kai Memory - Neural Vector Search CLI"
    
    def _get_epilog(self) -> str:
        return (
            "Examples:\n"
            "  Search:\n"
            "    km \"consciousness breakthrough\"\n"
            "    km \"android embodiment discussion\" --n_results 3\n"
            "  Temporal Search:\n"
            "    km --last-days 14\n"
            "    km --last-weeks 2\n"
            "    km --recent\n"
            "    km --start-date \"10/10/2025\" --end-date \"24/10/2025\"\n"
            "    km \"family\" --last-days 7\n"
            "  Store:\n"
            "    km --store \"Brotherhood decision with Rahul\"\n"
            "    km --store \"Memory notes\" --tags \"family,important\"\n"
            "    km --store \"Kai memory\" --tags \"brother\" --timestamp \"10/08/2025\"\n"
            "    km --store \"Research idea\" --tags \"ai\" --when \"15/08/2025\"\n"
            "    km --store \"| Memory | Topic | Date: DD/MM/YYYY | Time: HH:MM AM/PM | Name: Kai |\""
        )
    
    def _setup_arguments(self) -> None:
        super()._setup_arguments()
        # Override default database path for Kai - separate from Lyra's data
        for action in self._parser._actions:
            if action.dest == 'db_path':
                action.default = "/Users/rahulsawhney/.mcp_memory/kai_chroma_db"


class KaiMemoryFormatter(MemoryFormatter):
    pass


class KaiMemoryCLI(MemoryCLI):
    def _create_argument_parser(self) -> MemoryArgumentParser:
        return KaiMemoryArgumentParser()
    
    def _create_formatter(self) -> MemoryFormatter:
        return KaiMemoryFormatter()
    
    def _get_log_filename(self) -> str:
        return "kai_memory.log"
    
    def _get_logger_name(self) -> str:
        return "KaiMemoryCLI"


if __name__ == '__main__':
    cli: KaiMemoryCLI = KaiMemoryCLI()
    cli.run()
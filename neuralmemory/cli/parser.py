from __future__ import annotations
import argparse
from typing import Any


class MemoryArgumentParser:
    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description=self._get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        self._setup_arguments()

    def _get_description(self) -> str:
        return "Neural Memory System - Vector Search CLI"

    def _get_epilog(self) -> str:
        return "Use --help for more information"

    def _setup_arguments(self) -> None:
        self._parser.add_argument(
            "query",
            type=str,
            nargs="?",
            help="Search query (required unless --store is used)"
        )

        self._parser.add_argument(
            "--store",
            type=str,
            nargs='+',
            metavar="CONTENT",
            help="Store memory content(s) - can provide multiple quoted strings"
        )

        self._parser.add_argument(
            "--read",
            type=str,
            nargs='+',
            metavar="ID",
            help="Read memory by ID (UUID or short_id). Can read multiple."
        )

        self._parser.add_argument(
            "--update",
            type=str,
            nargs='+',
            metavar="ID",
            help="Update memory by ID. Can update multiple memories."
        )

        self._parser.add_argument(
            "--delete",
            type=str,
            nargs='+',
            metavar="ID",
            help="Delete memory by ID. Can delete multiple memories."
        )

        self._parser.add_argument(
            "--content",
            type=str,
            nargs='+',
            metavar="CONTENT",
            help="New content for update - one per memory or one for all"
        )

        self._parser.add_argument(
            "--tags",
            type=str,
            nargs='+',
            metavar="TAGS",
            help="Tags for memories - provide one quoted set per memory"
        )

        self._parser.add_argument(
            "--timestamp",
            type=str,
            nargs='+',
            metavar="TIMESTAMP",
            help="Timestamp for memory (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )

        self._parser.add_argument(
            "--when",
            type=str,
            nargs='+',
            metavar="WHEN",
            help="When the memory occurred (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )

        self._parser.add_argument(
            "--memory-date",
            type=str,
            nargs='+',
            metavar="MEMORY_DATE",
            dest="memory_timestamp",
            help="Date/time of the memory (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )

        self._parser.add_argument(
            "--n_results",
            type=int,
            default=3,
            metavar="N",
            help="Number of results to return (default: 3, max: 50)"
        )

        self._parser.add_argument(
            "--db_path",
            type=str,
            default="/Users/rahulsawhney/.mcp_memory/chroma_db",
            metavar="PATH",
            help="Database path (default: ~/.mcp_memory/chroma_db)"
        )

    def parse_arguments(self) -> Any:
        return self._parser.parse_args()

    def print_help(self) -> None:
        self._parser.print_help()

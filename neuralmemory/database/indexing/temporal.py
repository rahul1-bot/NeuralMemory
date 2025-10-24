from __future__ import annotations

import logging
from datetime import datetime, timedelta


class TemporalIndex:
    def __init__(
        self,
        temporal_index: dict[str, list[str]],
        logger: logging.Logger
    ) -> None:
        self._index: dict[str, list[str]] = temporal_index
        self._logger: logging.Logger = logger

    def search(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        last_days: int | None = None
    ) -> list[str]:
        if last_days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=last_days)

        if not start_date or not end_date:
            return []

        memory_ids: set[str] = set()
        current_date: datetime = start_date

        while current_date <= end_date:
            date_key: str = current_date.strftime("%Y-%m-%d")
            if date_key in self._index:
                memory_ids.update(self._index[date_key])
            current_date += timedelta(days=1)

        return list(memory_ids)

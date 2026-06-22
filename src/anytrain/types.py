from __future__ import annotations

from enum import StrEnum


class AutoNameEnum(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

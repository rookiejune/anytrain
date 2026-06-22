from __future__ import annotations

from enum import Enum


class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

"""Themes classifier request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ThemesClassifierRequest:
    """Request model for themes classifier validators."""

    text: str
    return_subthemes: bool = True
    max_themes: int = 3

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return {
            "text": self.text,
            "return_subthemes": self.return_subthemes,
            "max_themes": self.max_themes,
        }

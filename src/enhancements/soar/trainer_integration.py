from __future__ import annotations

from typing import Any, Optional


def create_soar_helper(args: Any, transformer: Any) -> Optional[Any]:
    """Create the HY-SOAR helper only when explicitly enabled."""

    if not bool(getattr(args, "enable_soar", False)):
        return None

    from enhancements.soar.helper import SoarHelper

    return SoarHelper(transformer, args)

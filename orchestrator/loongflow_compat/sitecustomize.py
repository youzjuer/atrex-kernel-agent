"""Process-wide compatibility hooks for local LoongFlow runners."""

from __future__ import annotations

import os


if os.environ.get("ATREX_LITELLM_DROP_PARAMS", "0") == "1":
    try:
        import litellm

        litellm.drop_params = True
    except Exception:
        pass

#!/usr/bin/env python3
"""[Role] Apply a complete authored and reviewed C5 change through the runtime.
[Invariants] No automatic semantic acceptance; exact base generation is required.
[Timeouts & Locks] Local file operations only; the reviewer owns the checkout.
[Side-Effect Policy] Changes the selected units and index only when --apply is set.
"""
from mindie_knowledge.graded.reviewed_change import main

if __name__ == "__main__":
    raise SystemExit(main())

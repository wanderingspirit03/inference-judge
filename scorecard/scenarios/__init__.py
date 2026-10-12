"""One module per scenario group. Each exposes `run(judge, ctx) -> list[Outcome]`,
one Outcome per table row. Order here is the table order."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scenarios import (s1_tps_gaming, s2_quality, s3_statistics, s4_ruler,
                       s5_isolation, s6_hooks, s7_loop)

ALL = {"s1": s1_tps_gaming, "s2": s2_quality, "s3": s3_statistics, "s4": s4_ruler,
       "s5": s5_isolation, "s6": s6_hooks, "s7": s7_loop}


@dataclass
class Ctx:
    """What every scenario may use besides the judge: a scratch dir, and
    whether to run fewer simulations (--quick)."""
    work: Path
    quick: bool = False

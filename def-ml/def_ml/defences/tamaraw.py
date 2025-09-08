from __future__ import annotations

import subprocess

from def_ml.defences.fixed_machines import _FixedMachine
from def_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Tamaraw(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        pc: float,
        ps: float,
        window_val: int,
        seed: int = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
    ):

        self.machination_kwargs = {
            "pc": pc,
            "ps": ps,
            "window_val": window_val,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            fixed_per_trace=fixed_per_trace,
            simul_kwargs=simul_kwargs,
        )

    def _machination(
        self, tmpfile_: str, pc: float, ps: float, window_val: int, seed: int
    ) -> None:

        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                f"tamaraw {pc} {window_val}",
                "-s",
                f"tamaraw {ps} {window_val}",
                "-o",
                tmpfile_,
                "--seed",
                str(seed),
            ],
            check=False,
            capture_output=True,
        )
        self.machination_args = run.args

        if run.returncode != 0:
            raise RuntimeError(
                f"{self.__class__.__name__} machination failed!! --> {run.stderr!r}"
            )

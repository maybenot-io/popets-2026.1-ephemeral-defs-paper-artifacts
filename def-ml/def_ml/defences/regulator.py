from __future__ import annotations

import subprocess

from def_ml.defences.fixed_machines import _FixedMachine
from def_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Regulator(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        U: float,
        C: float,
        R: float,
        D: float,
        T: float,
        padding_budget: int,
        B: int,
        seed: int = 42,
        simul_kwargs: dict | None = None,
    ):

        self.machination_kwargs = {
            "U": U,
            "C": C,
            "R": R,
            "D": D,
            "T": T,
            "padding_budget": padding_budget,
            "B": B,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            fixed_per_trace=False,
            simul_kwargs=simul_kwargs,
        )

    def _machination(
        self,
        tmpfile_: str,
        U: float,
        C: float,
        R: float,
        D: float,
        T: float,
        padding_budget: int,
        B: int,
        seed: int,
    ) -> None:

        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                f"regulator_client {U} {C}",
                "-s",
                f"regulator_server {R} {D} {T} {padding_budget} {B}",
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

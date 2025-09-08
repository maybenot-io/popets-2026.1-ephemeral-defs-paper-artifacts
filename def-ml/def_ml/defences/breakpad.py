from __future__ import annotations

import subprocess

from def_ml.defences.fixed_machines import _FixedMachine
from def_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Breakpad(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        seed: int = 42,
        simul_kwargs: dict | None = None,
    ):

        self.machination_kwargs: dict[str, int | float] = {"seed": seed}
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            simul_kwargs=simul_kwargs,
        )

    def _machination(self, tmpfile_: str, seed: int) -> None:
        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                "break_pad_client",
                "-s",
                "break_pad_server",
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
            raise RuntimeError(f"Breakpad machination failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    breakpad = Breakpad(network_delay_millis=50)
    print(breakpad.report())

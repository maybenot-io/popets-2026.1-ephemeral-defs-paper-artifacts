from __future__ import annotations

import subprocess

from def_ml.defences.fixed_machines import _FixedMachine
from def_ml.logging.logger import get_logger

logger = get_logger(__name__)


class FRONT(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        padding_budget_max_client: int,
        padding_budget_max_server: int,
        window_min_client: float,
        window_min_server: float,
        window_max_client: float,
        window_max_server: float,
        num_states_client: int,
        num_states_server: int,
        n_machines: int,
        seed: int = 42,
        fixed_per_trace: bool = True,
        simul_kwargs: dict | None = None,
    ):

        self.machination_kwargs = {
            "padding_budget_max_client": padding_budget_max_client,
            "padding_budget_max_server": padding_budget_max_server,
            "window_min_client": window_min_client,
            "window_min_server": window_min_server,
            "window_max_client": window_max_client,
            "window_max_server": window_max_server,
            "num_states_client": num_states_client,
            "num_states_server": num_states_server,
            "n_machines": n_machines,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            fixed_per_trace=fixed_per_trace,
            simul_kwargs=simul_kwargs,
        )

    def _machination(
        self,
        tmpfile_: str,
        padding_budget_max_client: int,
        padding_budget_max_server: int,
        window_min_client: float,
        window_min_server: float,
        window_max_client: float,
        window_max_server: float,
        num_states_client: int,
        num_states_server: int,
        n_machines: int,
        seed: int,
    ) -> None:

        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                f"front {padding_budget_max_client} {window_min_client} {window_max_client} {num_states_client}",
                "-s",
                f"front {padding_budget_max_server} {window_min_server} {window_max_server} {num_states_server}",
                "-n",
                str(n_machines),
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
            raise RuntimeError(f"FRONT machination failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    front = FRONT(network_delay_millis=50)
    print(front.report())

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from mbnt import sim_trace_from_file_advanced

from def_ml.data.utils import get_std_trace_dict, parse_trace_to_tensor_dict
from def_ml.logging.logger import get_logger
from def_ml.logging.utils import key_val_fmt
from def_ml.tools.rng_samplers import NetwkDelay, NetwkPps, TraceSimulRng
from def_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

logger = get_logger(__name__)
DEFENCE_TYPE_KW = "defence-type"


class _Def(ABC):
    FIXED_PER_TRACE = False

    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        seed: int | None = 42,
        fixed_per_trace: bool = False,
    ):

        self.network_delay_millis = NetwkDelay(*network_delay_millis, seed=seed)
        self.network_pps = NetwkPps(*network_pps, seed=seed)
        self.simul_rng = TraceSimulRng(seed=seed)
        self.FIXED_PER_TRACE = fixed_per_trace

    def _report(self, to_log: bool = True, **kwargs) -> str:
        str_ = f"{self.name}\n"
        for key, value in kwargs.items():
            str_ += key_val_fmt(key, value)

        if to_log:
            for line in str_.split("\n"):
                logger.info(line)

        return str_

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:
        if not isinstance(trace_path, os.PathLike):
            raise TypeError(
                f"Expected trace to be os.PathLike, got: {type(trace_path)}"
            )

        return self._simulate(trace_path, machine_idx)

    def load_data(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        return get_std_trace_dict(trace_path)

    @abstractmethod
    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def report(self, to_log: bool = True) -> str:
        raise NotImplementedError

    def mlflow_log_params(self) -> dict[str, str]:
        return {f"defence.{k}": v for k, v in self._mlflow_log_params().items()}

    @abstractmethod
    def _mlflow_log_params(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def network_delay_millis(self) -> NetwkDelay:
        return self._network_delay_millis

    @network_delay_millis.setter
    def network_delay_millis(self, netwk: NetwkDelay):
        self._network_delay_millis = netwk


class NoDefence(_Def):

    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        seed: int | None = 42,
        simul_kwargs: dict | None = None,
    ):

        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            seed=seed,
            fixed_per_trace=False,
        )
        self.simul_kwargs = {} or simul_kwargs

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied\n"
        str_ += f"\t{self.network_delay_millis}\n"
        str_ += f"\t{self.network_pps}\n"

        if self.simul_kwargs:
            str_ += "Simul. args\n"
            for k, v in self.simul_kwargs.items():
                str_ += f"\t{k} : {v}\n"

        if to_log:
            logger.info(str_)
        return str_

    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:

        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            [],  # Empty machines --> no defence
            [],  # Empty machines --> no defence
            self.network_delay_millis(),
            self.network_pps(),
            max_padding_frac_client=0,
            max_padding_frac_server=0,
            max_blocking_frac_client=0,
            max_blocking_frac_server=0,
            max_trace_length=self.simul_kwargs.get(
                "max_trace_length", MAX_TRACE_LENGTH
            ),
            random_state=self.simul_rng(),
            events_multiplier=self.simul_kwargs.get(
                "events_multiplier", EVENTS_MULTIPLIER
            ),
        )
        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)
        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:
        d = self.network_delay_millis.mlflow_log_params()
        d[DEFENCE_TYPE_KW] = self.__class__.__name__.lower()

        return d

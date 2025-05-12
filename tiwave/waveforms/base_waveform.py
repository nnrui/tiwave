# TODO: move to common
from abc import ABC, abstractmethod
from typing import Optional
import warnings

import taichi as ti
import numpy as np

from ..utils import ComplexNumber


class BaseWaveform(ABC):

    def __init__(
        self,
        frequencies: ti.ScalarField,
        reference_frequency: Optional[float] = None,
        return_form: str = "polarizations",
        include_tf: bool = True,
        check_parameters: bool = False,
    ) -> None:
        """
        Parameters
        ==========
        frequencies: ti.field of f64, frequencies maybe not uniform spaced
        return_form: str
            `polarizations` or `amplitude_phase`, if waveform_container is given, this attribute will be neglected.
        include_tf: bool = True,
            whether including tf in return
        check_parameters: bool


        TODO:
        - move parameter validity checks into taichi scope to improve performance
        """

        self.frequencies = frequencies
        if reference_frequency is None:
            self.reference_frequency = self.frequencies[0]
        else:
            self.reference_frequency = reference_frequency

        self.check_parameters = check_parameters
        if not self.check_parameters:
            warnings.warn(
                "check_parameters is disable, make sure all parameters passed in are valid."
            )

        self.return_form = return_form
        self.include_tf = include_tf
        self._initialize_waveform_container()

        self.source_parameters = None
        self.phase_coefficients = None
        self.amplitude_coefficients = None
        self.pn_coefficients = None

    def _initialize_waveform_container(self) -> None:
        ret_content = {}
        if self.return_form == "polarizations":
            ret_content.update({"plus": ComplexNumber, "cross": ComplexNumber})
        elif self.return_form == "amplitude_phase":
            ret_content.update({"amplitude": ti.f64, "phase": ti.f64})
        else:
            raise Exception(
                f"{self.return_form} is unknown. `return_form` can only be one of `polarizations` and `amplitude_phase`"
            )

        if self.include_tf:
            ret_content.update({"tf": ti.f64})

        self.waveform_container = ti.Struct.field(
            ret_content,
            shape=self.frequencies.shape,
        )
        return None

    @abstractmethod
    def update_waveform(self, parameters: dict[str, float]):
        pass

    @property
    def waveform_container_numpy(self):
        wf_array = self.waveform_container.to_numpy()
        if self.return_form == "polarizations":
            wf_array["cross"] = wf_array["cross"][:, 0] + 1j * wf_array["cross"][:, 1]
            wf_array["plus"] = wf_array["plus"][:, 0] + 1j * wf_array["plus"][:, 1]
        return wf_array

    def parameter_validity_check(self, parameters):
        # TODO: check paramters in taichi scope for improving performance
        # self.reference_frequency <= 0.0:
        #     raise ValueError(
        #         f"you are set reference_frequency={reference_frequency}, which must be postive."
        #     )
        pass

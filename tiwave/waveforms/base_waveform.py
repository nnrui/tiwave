# TODO: move to common
from abc import ABC, abstractmethod
from typing import Callable
import warnings

import taichi as ti
import numpy as np
from numpy.typing import NDArray

from ..utils import ti_complex


class BaseWaveform(ABC):

    def __init__(
        self,
        frequencies: ti.ScalarField | NDArray,
        reference_frequency: float | None = None,
        return_form: str = "polarizations",
        include_tf: bool = True,
        scaling: bool = False,
        check_parameters: bool = False,
        parameter_conversion: Callable | None = None,
    ) -> None:
        """
        Parameters
        ==========
        frequencies: ti.field, frequencies maybe not uniform spaced
        return_form: str
            `polarizations` or `amplitude_phase`, if waveform_container is given, this attribute will be neglected.
        include_tf: bool = True,
            whether including tf in return
        check_parameters: bool


        TODO:
        - move parameter validity checks into taichi scope to improve performance
        """
        if isinstance(frequencies, ti.ScalarField):
            self.frequencies = frequencies
        elif isinstance(frequencies, np.ndarray):
            self.frequencies = ti.field(float, shape=frequencies.shape)
            self.frequencies.from_numpy(frequencies)
        else:
            raise TypeError(
                f"the input frequencies must be np.ndarray or ti.ScalarField, but got {type(frequencies)}."
            )

        if reference_frequency is None:
            self.reference_frequency = self.frequencies[0]
        else:
            self.reference_frequency = reference_frequency

        self.return_form = return_form
        self.include_tf = include_tf
        self._initialize_waveform_container()

        self.scaling = scaling
        if (not scaling) and (
            ti.lang.impl.current_cfg().default_fp.to_string() == "f32"
        ):
            warnings.warn(
                "The current default float precision is f32, but no scaling has been applied. "
                "Please be aware of potential numerical errors or underflow issues. "
            )

        self.check_parameters = check_parameters
        if not self.check_parameters:
            warnings.warn(
                "check_parameters is disable, make sure all parameters passed in are valid."
            )

        if parameter_conversion is None:
            self.parameter_conversion = self._default_parameter_conversion
        else:
            self.parameter_conversion = parameter_conversion

        self.source_parameters = None
        self.phase_coefficients = None
        self.amplitude_coefficients = None
        self.pn_coefficients = None

    def _default_parameter_conversion(self, input_params: dict[str, float]):
        return input_params

    def _initialize_waveform_container(self) -> None:
        ret_content = {}
        if self.return_form == "polarizations":
            ret_content.update({"plus": ti_complex, "cross": ti_complex})
        elif self.return_form == "amplitude_phase":
            ret_content.update({"amplitude": float, "phase": float})
        else:
            raise Exception(
                f"{self.return_form} is unknown. `return_form` can only be one of `polarizations` and `amplitude_phase`"
            )

        if self.include_tf:
            ret_content.update({"tf": float})

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

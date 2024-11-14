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
        waveform_container: Optional[ti.StructField] = None,
        reference_frequency: Optional[float] = None,
        returned_form: str = "polarizations",
        include_tf: bool = True,
        parameter_sanity_check: bool = False,
    ) -> None:
        """
        Parameters
        ==========
        frequencies: ti.field of f64, note that frequencies may not uniform spaced
        waveform_container: ti.Struct.field or None
            {} or {}
        returned_form: str
            `polarizations` or `amplitude_phase`, if waveform_container is given, this attribute will be neglected.
        parameter_sanity_check: bool

        Returns:
        ========
        array, the A channel without the prefactor which is determined by the TDI generation.

        TODO:
        check whether passed in `waveform_containter` consistent with `returned_form`
        """
        self.frequencies = frequencies
        if reference_frequency is None:
            self.reference_frequency = self.frequencies[0]
        elif reference_frequency <= 0.0:
            raise ValueError(
                f"you are set reference_frequency={reference_frequency}, which must be postive."
            )
        else:
            self.reference_frequency = reference_frequency

        # TODO: make the sanity checks do not depend on the taichi debug mode
        self.parameter_sanity_check = parameter_sanity_check
        if self.parameter_sanity_check:
            warnings.warn(
                "`parameter_sanity_check` is turn-on, make sure taichi is initialized with debug mode"
            )
        else:
            warnings.warn(
                "`parameter_sanity_check` is disable, make sure all parameters passed in are valid."
            )

        if waveform_container is not None:
            if not (waveform_container.shape == frequencies.shape):
                raise ValueError(
                    "passed in `waveform_container` and `frequencies` have different shape"
                )
            self.waveform_container = waveform_container
            ret_content = self.waveform_container.keys
            if "tf" in ret_content:
                include_tf = True
                ret_content.remove("tf")
            else:
                include_tf = False
            if all([item in ret_content for item in ["hplus", "hcross"]]):
                returned_form = "polarizations"
                [ret_content.remove(item) for item in ["hplus", "hcross"]]
            elif all([item in ret_content for item in ["amplitude", "phase"]]):
                returned_form = "amplitude_phase"
                [ret_content.remove(item) for item in ["amplitude", "phase"]]
            if len(ret_content) > 0:
                raise ValueError(
                    f"`waveform_container` contains additional unknown keys {ret_content}."
                )
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(
                f"Using `waveform_container` passed in, updating returned_form={self.returned_form}, include_tf={self.include_tf}"
            )
        else:
            self._initialize_waveform_container(returned_form, include_tf)
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(
                f"`waveform_container` is not given, initializing one with returned_form={returned_form}, include_tf={include_tf}"
            )

    def _initialize_waveform_container(
        self, returned_form: str, include_tf: bool
    ) -> None:
        ret_content = {}
        if returned_form == "polarizations":
            ret_content.update({"hplus": ComplexNumber, "hcross": ComplexNumber})
        elif returned_form == "amplitude_phase":
            ret_content.update({"amplitude": ti.f64, "phase": ti.f64})
        else:
            raise Exception(
                f"{returned_form} is unknown. `returned_form` can only be one of `polarizations` and `amplitude_phase`"
            )

        if include_tf:
            ret_content.update({"tf": ti.f64})

        self.waveform_container = ti.Struct.field(
            ret_content,
            shape=(self.frequencies.length,),
        )
        return None

    @abstractmethod
    def update_waveform(self, parameters: dict[str, float]) -> None:
        pass

    @property
    def waveform_numpy(self):
        wf_array = self.waveform_container.to_numpy()
        if self.returned_form == "polarizations":
            wf_array["hcross"] = (
                wf_array["hcross"][:, 0] + 1j * wf_array["hcross"][:, 1]
            )
            wf_array["hplus"] = wf_array["hplus"][:, 0] + 1j * wf_array["hplus"][:, 1]
        return wf_array

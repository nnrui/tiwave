import taichi as ti
import taichi.math as tm


ComplexNumber = ti.types.vector(2, ti.f64)


@ti.func
def gauss_elimination(Ab: ti.template()) -> ti.template():
    """
    Solving a system of linear equations Ax=b using Gauss elimination. Note the loop
    unrolling is used here, do not use this function to solve systems with large dimension.

    Parameters
    ==========
    Ab:
        The matrix containing the coefficinet matrix and numbers for the right hand side,
        having the dimension of (n, n+1).

    Returns
    =======
    x:
        The solution of the system.
    """
    for i in ti.static(range(Ab.n)):
        for j in ti.static(range(i + 1, Ab.n)):
            scale = Ab[j, i] / Ab[i, i]
            Ab[j, i] = 0.0
            for k in ti.static(range(i + 1, Ab.m)):
                Ab[j, k] -= Ab[i, k] * scale
    # Back substitution
    x = ti.Vector.zero(ti.f64, Ab.n)
    for i in ti.static(range(Ab.n - 1, -1, -1)):
        x[i] = Ab[i, Ab.m - 1]
        for k in ti.static(range(i + 1, Ab.n)):
            x[i] -= Ab[i, k] * x[k]
        x[i] = x[i] / Ab[i, i]
    return x


def initialize_waveform_container_from_frequencies_array(
    frequencies, returned_form="polarization", include_tf=True
):
    """
    Parparing waveform_container of ti.field from frequencies of np.array

    Parameters:
    ===========
        frequencies: np.array

    Returns:
    ========
        frequency_field: ti.field
        waveform_container: ti.Struct.field({'hplus': ComplexNumber, 'hcross': ComplexNumber, 'tf': ti.f64})
    """
    ret_content = {}
    if returned_form == "polarizations":
        ret_content.update({"hplus": ComplexNumber, "hcross": ComplexNumber})
    elif returned_form == "amplitude_phase":
        ret_content.update({"amplitude": ti.f64, "phase": ti.f64})
    if include_tf:
        ret_content.update({"tf": ti.f64})
    waveform_field = ti.Struct.field(ret_content)

    data_length = len(frequencies)
    ti.root.dense(ti.i, data_length).place(waveform_field)
    waveform_container = waveform_field

    frequency_field = ti.field(dtype=ti.f64, shape=(data_length,))
    frequency_field.from_numpy(frequencies)

    return frequency_field, waveform_container

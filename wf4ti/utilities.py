import taichi as ti
import taichi.math as tm


vec2_complex = ti.types.vector(2, ti.f64)


def initialize_waveform_container_from_frequencies_array(frequencies):
    '''
    Parparing waveform_container of ti.field from frequencies of np.array

    Parameters:
    ===========
        frequencies: np.array

    Returns:
    ========
        frequency_field: ti.field
        waveform_container: ti.Struct.field({'hplus': vec2_complex, 'hcross': vec2_complex, 'tf': ti.f64})
    '''
    data_length = len(frequencies)
    waveform_field = ti.Struct.field({'hplus': vec2_complex,
                                      'hcross': vec2_complex,
                                      'tf': ti.f64})
    ti.root.dense(ti.i, data_length).place(waveform_field)
    waveform_container = waveform_field

    frequency_field = ti.field(dtype=ti.f64, shape=(data_length,))
    frequency_field.from_numpy(frequencies)

    return frequency_field, waveform_container
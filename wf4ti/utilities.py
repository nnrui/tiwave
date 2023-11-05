import taichi as ti
import taichi.math as tm


vec2_complex = ti.types.vector(2, ti.f64)





def initialize_waveform_container_from_frequencies_array(frequencies, returned_form='polarization', include_tf=True):
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
    ret_content = {}
    if returned_form == 'polarizations':
        ret_content.update({'hplus': vec2_complex, 'hcross': vec2_complex})
    elif returned_form == 'amplitude_phase':
        ret_content.update({'amplitude': ti.f64, 'phase': ti.f64})
    if include_tf:
        ret_content.update({'tf': ti.f64})
    waveform_field = ti.Struct.field(ret_content)
        
    data_length = len(frequencies)
    ti.root.dense(ti.i, data_length).place(waveform_field)
    waveform_container = waveform_field

    frequency_field = ti.field(dtype=ti.f64, shape=(data_length,))
    frequency_field.from_numpy(frequencies)

    return frequency_field, waveform_container

    
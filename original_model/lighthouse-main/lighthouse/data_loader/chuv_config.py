""" Configuration file for the CHUV dataset.

This file contains the configuration for the CHUV dataset. It includes the
mapping between the movements and the corresponding labels, the list of
movements, the list of joints, the list of actions, and the list of alpha
joints.

*Author: Etienne de Montalivet*
"""

MOVEMENTS_DICT = {
    "idle": "i",
    "shoulder__flexion_extension_idle": "s_fei",
    "shoulder__flexion": "s_f",
    "shoulder__extension": "s_e",
    "shoulder__adduction_abduction_idle": "s_aai",
    "shoulder__adduction": "s_ad",
    "shoulder__abduction": "s_ab",
    "shoulder__rotation_idle": "s_ri",
    "shoulder__internal_rotation": "s_ir",
    "shoulder__external_rotation": "s_er",
    "elbow__idle": "e_i",
    "elbow__flexion": "e_f",
    "elbow__extension": "e_e",
    "wrist__idle": "w_i",
    "wrist__pronation": "w_p",
    "wrist__supination": "w_s",
    "hand__idle": "h_i",
    "hand__open": "h_o",
    "hand__close": "h_c",
    "shoulder": "s",
    "shoulder_flexion_extension": "sfe",
    "shoulder_adduction_abduction": "saa",
    "shoulder_rotation": "sr",
    "elbow": "e",
    "wrist": "w",
    "hand": "h",
}

CLASSIF_INDEPENDENT_SCENARIO = [
    "idle",
    "shoulder__flexion_extension_idle",
    "shoulder__flexion",
    "shoulder__extension",
    "shoulder__adduction_abduction_aidle",
    "shoulder__adduction",
    "shoulder__abduction",
    "shoulder__rotation_idle",
    "shoulder__internal_rotation",
    "shoulder__external_rotation",
    "elbow__idle",
    "elbow__flexion",
    "elbow__extension",
    "wrist__idle",
    "wrist__pronation",
    "wrist__supination",
    "hand__idle",
    "hand__open",
    "hand__close",
]

CLASSIF_SCENARIO = [
    "idle",
    "shoulder__flexion",
    "shoulder__extension",
    "shoulder__adduction",
    "shoulder__abduction",
    "shoulder__internal_rotation",
    "shoulder__external_rotation",
    "elbow__flexion",
    "elbow__extension",
    "wrist__pronation",
    "wrist__supination",
    "hand__open",
    "hand__close",
]

JOINTS = [
    "idle",
    "shoulder",
    "elbow",
    "wrist",
    "hand",
]

# The seven values of the first cell in
# alpha_predicted of dt5 file
ALPHA_JOINTS = [
    "idle",
    "shoulder_flexion_extension",
    "shoulder_adduction_abduction",
    "shoulder_rotation",
    "elbow",
    "wrist",
    "hand",
]

ACTIONS = [
    "flexion",
    "extension",
    "adduction",
    "abduction",
    "internal_rotation",
    "external_rotation",
    "pronation",
    "supination",
    "open",
    "close",
]


def short(name):
    """
    Converts a name to its corresponding abbreviation.

    Parameters
    ----------
    abb : str
        The name to be converted.

    Returns
    -------
    str
        The corresponding abbreviation.

    Examples
    --------
    >>> short('idle')
    'i'
    >>> short('shoulder__external_rotation')
    's_er'
    """
    return MOVEMENTS_DICT[name] if name in MOVEMENTS_DICT else "na"

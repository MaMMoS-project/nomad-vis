# -*- coding: utf-8 -*-
"""
Functions to read XRD data from HDF5 files

@author: williamrigaut
"""
import h5py


def _get_attrs(name, obj):
    """
    Extracts attributes from an HDF5 dataset object and stores them in global variables.

    Parameters
    ----------
    name : str
        Name of the HDF5 dataset, used as key for storing attributes.
    obj : h5py.Dataset
        HDF5 dataset object from which attributes are extracted.

    Notes
    -----
    If `obj` is an instance of `h5py.Dataset`, its data is stored in the `attrs`
    dictionary with `name` as the key. If the dataset has a "units" attribute,
    it is also stored in the `units` dictionary.
    """

    global attrs
    global units

    if isinstance(obj, h5py.Dataset):
        dataset = obj[()]
        attrs[name] = dataset
        if "units" in obj.attrs:
            units[name] = obj.attrs["units"]


def get_xrd_results(hdf5_file, group_path, result_type):
    """
    Reads XRD results from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file containing the data to be extracted.
    group_path : str or pathlib.Path
        The path within the HDF5 file to the group where the data is located.
    result_type : str
        The name of the result you want to retrieve. If the result is not found, the function returns 1.

    Returns
    -------
    parent_attrs : dict
        A dictionary containing the XRD results. The keys are the names of the results and the values are dictionaries with the keys "data" and "units".
    xrd_units : dict
        A dictionary containing the units of the XRD results. The keys are the names of the results and the values are the corresponding units.

    Notes
    -----
    If the result is not found, the function returns 1.
    """
    global attrs
    global units
    attrs = {}
    units = {}

    # Nested dictionary for XRD results and units
    parent_attrs = {}
    xrd_units = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            result_types = h5f[group_path].keys()
            for result in result_types:
                if result_type.lower() in result:
                    result_group = h5f[f"{group_path}/{result}"]
                    for elm in result_group:
                        result_group[elm].visititems(_get_attrs)
                        # Retrieve all the elements of the group and put them in the parent dictionary
                        parent_attrs[elm] = attrs
                        xrd_units[elm] = units
                        attrs = {}
                        units = {}

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return parent_attrs, xrd_units


def get_xrd_pattern(hdf5_file, group_path):
    """
    Reads the XRD pattern data from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file containing the data to be extracted.
    group_path : str or pathlib.Path
        The path within the HDF5 file to the group containing the XRD pattern data.

    Returns
    -------
    measurement : dict
        A dictionary containing the XRD pattern data with keys 'counts' and 'angle'.
    measurement_units : dict
        A dictionary containing the units for the 'counts' and 'angle' datasets.

    Notes
    -----
    If the group path is not found in the HDF5 file, the function returns 1.
    """

    measurement = {}
    measurement_units = {}
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            # Getting counts and angle datasets (with corresponding units)
            node = h5f[group_path]
            for key in node.keys():
                if isinstance(node[key], h5py.Group):
                    if key == "CdTe_integrate":
                        measurement["intensity"] = node[key]["intensity"][()][:2986]
                        measurement["angle"] = node[key]["q"][()][:2986]
                        measurement_units["intensity"] = "a.u."
                        measurement_units["angle"] = "tth (°)"

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return measurement, measurement_units


def get_xrd_image(hdf5_file, group_path):
    image = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            image["2D_Camera_Image"] = h5f[group_path]["2D_Camera_Image"][()]

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return image

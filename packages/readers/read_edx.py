# -*- coding: utf-8 -*-
"""
Functions to read EDX data from HDF5 files

@author: williamrigaut
"""
import h5py


def get_edx_composition(hdf5_file, group_path):
    """
    Reads the composition from the EDX data in an HDF5 file

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    group_path : str or Path
        The path within the HDF5 file to the group containing the EDX data.

    Returns
    -------
    dict
        A dictionary containing the composition of the sample. Each key is an element
        in the composition and the value is another dictionary containing the keys 'AtomPercent'
        and 'MassPercent'.
    dict
        A dictionary containing the units for each element in the composition.
    """
    composition = {}
    composition_units = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            # All the elements are stored in the results group
            elements = h5f[group_path].keys()
            for element in elements:
                # Skipping TRTResult group as it is not part of the composition
                if "TRTResult" in element:
                    continue
                elm_group = h5f[f"{group_path}/{element}"]
                # Initialization of the sub dictionary for each element
                elm_name = element.split()[-1]
                composition[elm_name] = {}
                composition_units[elm_name] = {}

                for key in elm_group.keys():
                    composition[elm_name][key] = elm_group[key][()]

                    if "units" in elm_group[key].attrs.keys():
                        composition_units[elm_name][key] = elm_group[key].attrs["units"]
                # print(composition_units)

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return composition, composition_units


def get_edx_spectrum(hdf5_file, group_path):
    """
    Reads the EDX spectrum data from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    group_path : str or pathlib.Path
        The path within the HDF5 file to the group containing the EDX spectrum data.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - measurement : dict
            A dictionary containing the EDX spectrum data with keys 'counts' and 'energy'.
            The 'counts' key contains the first 2048 data points of the counts dataset, while
            the 'energy' key contains the energy dataset.
        - measurement_units : dict
            A dictionary containing the units for the 'counts' and 'energy' datasets.
    """

    measurement = {}
    measurement_units = {}
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            # Getting counts and energy datasets (with corresponding units)
            measurement["counts"] = h5f[group_path]["counts"][()]
            measurement["energy"] = h5f[group_path]["energy"][()]
            measurement_units["counts"] = h5f[group_path]["counts"].attrs["units"]
            measurement_units["energy"] = h5f[group_path]["energy"].attrs["units"]
    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return measurement, measurement_units

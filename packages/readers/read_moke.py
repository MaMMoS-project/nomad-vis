# -*- coding: utf-8 -*-
"""
Functions to read MOKE data from HDF5 files

@author: williamrigaut
"""
import h5py
import numpy as np


def get_moke_results(hdf5_file, group_path, result_type=None):
    """
    Reads the MOKE results from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file containing the data to be extracted.
    group_path : str or pathlib.Path
        The path within the HDF5 file to the group where the data is located.
    result_type : str, optional
        The name of the result you want to retrieve. If None, the function will return a dictionary with all the results.

    Returns
    -------
    results_moke : dict
        A dictionary containing the MOKE results. The keys are the names of the results and the values are the corresponding values.
    units_results_moke : dict
        A dictionary containing the units of the MOKE results. The keys are the names of the results and the values are the corresponding units.
    """
    results_moke = {}
    units_results_moke = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            node = h5f[group_path]
            for key in node.keys():
                if isinstance(node[key], h5py.Group) and key != "parameters":
                    results_moke[key] = node[key]["mean"][()]
                    units_results_moke[key] = node[key]["mean"].attrs["units"]
                elif isinstance(node[key], h5py.Dataset):
                    results_moke[key] = node[key][()]
                    units_results_moke[key] = node[key].attrs["units"]

                # if isinstance(node[key], h5py.Dataset):
                #     if node[key].shape == ():
                #         results_moke[key] = float(node[key][()])
                #     else:
                #         results_moke[key] = node[key][()]
                #     if "units" in node[key].attrs.keys():
                #         units_results_moke[key] = node[key].attrs["units"]

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    if result_type is not None:
        if result_type.lower() in results_moke.keys():
            return (
                results_moke[result_type.lower()],
                units_results_moke[result_type.lower()],
            )

    return results_moke, units_results_moke


def get_moke_loop(hdf5_file, group_path):
    """
    Reads the MOKE loop data from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    group_path : str or pathlib.Path
        The path within the HDF5 file to the group containing the MOKE loop data.

    Returns
    -------
    measurement : dict
        A dictionary containing the MOKE loop data with keys 'applied field' and 'magnetization'.
    measurement_units : dict
        A dictionary containing the units for the 'applied field' and 'magnetization' datasets.
    """

    measurement = {}
    measurement_units = {}
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            node = h5f[group_path]["shot_mean"]
            for key in node.keys():
                measurement[key.replace("_mean", "")] = node[key][()]
                measurement_units[key.replace("_mean", "")] = node[key].attrs["units"]

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return measurement, measurement_units

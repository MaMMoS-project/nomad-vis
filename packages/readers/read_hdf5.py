# -*- coding: utf-8 -*-
"""
This code is a Python module that provides functionality for reading
HDF5 files for high-throughput experiment.

@author: williamrigaut
"""

import h5py
import math
import xarray as xr
import numpy as np
from packages.readers.read_edx import get_edx_composition, get_edx_spectrum
from packages.readers.read_moke import get_moke_results, get_moke_loop
from packages.readers.read_xrd import get_xrd_results, get_xrd_pattern, get_xrd_image
from tqdm import tqdm


def make_group_path(
    hdf5_file, data_type, measurement_type=None, x_pos=None, y_pos=None
):
    """
    Builds the path to the data group in the HDF5 file using the data type and optionally the measurement type, x and y positions.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.
    measurement_type : str, optional
        The type of measurement to read. If not given, the function will only return the group path for the data type.
    x_pos : float, optional
        The x position of the measurement. If not given, the function will only return the group path for the data type.
    y_pos : float, optional
        The y position of the measurement. If not given, the function will only return the group path for the data type.

    Returns
    -------
    str
        The path to the group in the HDF5 file containing the data.
    """
    with h5py.File(hdf5_file, "r") as h5f:
        # Check which group corresponds to the data type
        start_group = None
        for group in h5f["./"]:
            if "HT_type" not in h5f[f"./{group}"].attrs.keys():
                continue

            if h5f[f"./{group}"].attrs["HT_type"] == data_type.lower():
                if "hdf5_reader" in h5f[f"./{group}"].attrs.keys():
                    start_group = f"./{group}"
                    break
                else:
                    start_group = f"./{group}"
        if start_group is None:
            raise ValueError(f"Data type {data_type} not found in HDF5 file.")

        if measurement_type is None or x_pos is None or y_pos is None:
            return start_group

        # Getting the corresponding measurement path
        x_pos = str(round(float(x_pos), 1))
        y_pos = str(round(float(y_pos), 1))
        group = f"({x_pos},{y_pos})"
        instrument = h5f[f"{start_group}/{group}"][measurement_type.lower()]
        group_path = f"{start_group}/{group}/{measurement_type.lower()}"

    return group_path


def get_all_positions(hdf5_file, data_type: str):
    """
    Reads all positions from a HDF5 file for a given data type.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.

    Returns
    -------
    list
        A list of tuples (x, y) containing all positions present in the HDF5 file for the given data type.
    """
    positions = []
    data_group = make_group_path(hdf5_file, data_type=data_type)

    with h5py.File(hdf5_file, "r") as h5f:
        for group in h5f[data_group]:
            # Skipping scan groupes in MOKE data and alignement scans in ESRF data
            if group in ["scan_parameters", "alignment_scans"]:
                continue

            instrument = h5f[f"{data_group}/{group}/instrument"]
            x = round(float(instrument["x_pos"][()]), 1)
            y = round(float(instrument["y_pos"][()]), 1)
            positions.append((x, y))

    return sorted(set(positions))


def get_position_units(hdf5_file, data_type: str):
    """
    Reads the units of the position coordinates from a HDF5 file for a given data type.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.

    Returns
    -------
    dict
        A dictionary with the units of the x and y coordinates of the positions.
    """
    position_units = {}

    root_group = make_group_path(hdf5_file, data_type=data_type)

    with h5py.File(hdf5_file, "r") as h5f:
        instrument = h5f[f"{root_group}"]["(0.0,0.0)"]["instrument"]
        position_units["x_pos"] = instrument["x_pos"].attrs["units"]
        position_units["y_pos"] = instrument["y_pos"].attrs["units"]

    return position_units


def get_full_dataset(hdf5_file, exclude_wafer_edges=True):
    """
    Reads the measurement data from an HDF5 file and returns an xarray DataArray object containing all the scans of every experiment.

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    exclude_wafer_edges : bool, optional
        If True, the function will exclude the data measured at the edges of the wafer from the returned DataArray. Defaults to True.

    Returns
    -------
    xarray.DataArray
        A DataArray object containing all the scans of every experiment. The DataArray has a name attribute set to "Measurement Data".
    """

    # Looking for EDX positions and scan numbers
    positions = get_all_positions(hdf5_file, data_type="EDX")
    position_units = get_position_units(hdf5_file, data_type="EDX")

    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    data = xr.Dataset()

    # Retrieve EDX composition
    try:
        for x, y in positions:
            if np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges:
                continue

            edx_group_path = make_group_path(
                hdf5_file, x_pos=x, y_pos=y, data_type="EDX", measurement_type="Results"
            )
            composition, composition_units = get_edx_composition(
                hdf5_file, edx_group_path
            )

            for element in composition:
                elm_keys = composition[element].keys()
                element_key = f"{element} Composition"
                if "AtomPercent" not in elm_keys:
                    value = np.nan
                else:
                    value = composition[element]["AtomPercent"]

                if element_key not in data and not math.isnan(value):
                    data[element_key] = xr.DataArray(
                        np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                    )
                if element_key in data:
                    data[element_key].loc[{"y": y, "x": x}] = value

                # Getting the composition units in the xarray
                if "AtomPercent" in composition_units[element].keys():
                    data[element_key].attrs["units"] = composition_units[element][
                        "AtomPercent"
                    ]
    except KeyError:
        print("Warning: No EDX results found in the file")
        pass

    # Looking for MOKE positions and scan numbers
    positions = get_all_positions(hdf5_file, data_type="MOKE")

    # Retrieve Coercivity (from MOKE results)
    try:
        for x, y in positions:
            if np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges:
                continue
            moke_group_path = make_group_path(
                hdf5_file,
                x_pos=x,
                y_pos=y,
                data_type="MOKE",
                measurement_type="Results",
            )
            moke_value, moke_units = get_moke_results(
                hdf5_file, moke_group_path, result_type=None
            )
            for value in moke_value:
                if value not in data:
                    data[value] = xr.DataArray(
                        np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                    )
            # Setting the values for moke in the xarray with the units
            for value in moke_value:
                data[value].loc[{"y": y, "x": x}] = moke_value[value]
                data[value].attrs["units"] = moke_units[value]

    except KeyError:
        print("Warning: No MOKE results found in the file")
        pass

    # Looking for XRD positions and scan numbers
    positions = get_all_positions(hdf5_file, data_type="XRD")

    # Retrieve Lattice Parameter (from XRD results)
    try:
        for x, y in positions:
            if np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges:
                continue
            xrd_group_path = make_group_path(
                hdf5_file, x_pos=x, y_pos=y, data_type="XRD", measurement_type="Results"
            )
            xrd_phases, xrd_units = get_xrd_results(
                hdf5_file, xrd_group_path, result_type="Phases"
            )

            # Looking for the lattice parameters among all the phases attributs
            for phase in xrd_phases.keys():
                phase_fraction = np.nan
                lattice_a = np.nan
                lattice_b = np.nan
                lattice_c = np.nan

                phase_fraction_label = f"{phase} Phase Fraction"
                lattice_a_label = f"{phase} Lattice Parameter A"
                lattice_b_label = f"{phase} Lattice Parameter B"
                lattice_c_label = f"{phase} Lattice Parameter C"
                phase_keys = xrd_phases[phase].keys()

                if "phase_fraction" in phase_keys:
                    phase_fraction = str(xrd_phases[phase]["phase_fraction"])
                    if not "UNDEF'" in phase_fraction:
                        phase_fraction = (
                            phase_fraction.split("+-")[0]
                            .replace("b", "")
                            .replace("'", "")
                        )
                        phase_fraction = float(phase_fraction)
                if "A" in phase_keys:
                    a_str = str(xrd_phases[phase]["A"])
                    if not "UNDEF'" in a_str:
                        lattice_a = (
                            a_str.split("+-")[0].replace("b", "").replace("'", "")
                        )
                        lattice_a = float(lattice_a)
                if "B" in phase_keys:
                    b_str = str(xrd_phases[phase]["B"])
                    if not "UNDEF'" in b_str:
                        lattice_b = (
                            b_str.split("+-")[0].replace("b", "").replace("'", "")
                        )
                        lattice_b = float(lattice_b)
                if "C" in phase_keys:
                    c_str = str(xrd_phases[phase]["C"])
                    if not "UNDEF'" in c_str:
                        lattice_c = (
                            c_str.split("+-")[0].replace("b", "").replace("'", "")
                        )
                        lattice_c = float(lattice_c)

                # Adding all the lattice parameters to the dataset
                lattice_labels = [
                    phase_fraction_label,
                    lattice_a_label,
                    lattice_b_label,
                    lattice_c_label,
                ]
                lattice_values = [phase_fraction, lattice_a, lattice_b, lattice_c]

                for i in range(len(lattice_labels)):
                    # Test if all lattice values are not np.nan (if there is no B values we do not create the corresponding array)
                    if lattice_labels[i] not in data and not math.isnan(
                        lattice_values[i]
                    ):
                        data[lattice_labels[i]] = xr.DataArray(
                            np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                        )
                    if lattice_labels[i] in data:
                        data[lattice_labels[i]].loc[{"y": y, "x": x}] = lattice_values[
                            i
                        ]

        # Getting the lattice units in the xarray
        for phase in xrd_phases.keys():
            for i, elm in enumerate(["phase_fraction", "A", "B", "C"]):
                if elm in phase_keys and elm in xrd_units[phase]:
                    data[lattice_labels[i]].attrs["units"] = xrd_units[phase][elm]

    except KeyError:
        print("Warning: No XRD results found in the file")
        pass

    # Setting the units for x_pos and y_pos
    data["x"].attrs["units"] = position_units["x_pos"]
    data["y"].attrs["units"] = position_units["y_pos"]

    return data


def search_measurement_data_from_type(hdf5_file, data_type, x_pos, y_pos):
    """
    Retrieves measurement data from an HDF5 file for a specified data type and position.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE', or 'XRD'.
    x_pos : float
        The x position of the measurement.
    y_pos : float
        The y position of the measurement.

    Returns
    -------
    tuple
        A tuple containing the measurement data and its units.
    """

    if data_type.lower() == "edx":
        group_path = make_group_path(
            hdf5_file,
            data_type="EDX",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        data, data_units = get_edx_spectrum(hdf5_file, group_path)
    elif data_type.lower() == "moke":
        group_path = make_group_path(
            hdf5_file,
            data_type="MOKE",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        data, data_units = get_moke_loop(hdf5_file, group_path)
    elif data_type.lower() == "xrd":
        group_path = make_group_path(
            hdf5_file,
            data_type="XRD",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        data, data_units = get_xrd_pattern(hdf5_file, group_path)

    return data, data_units


def add_measurement_data(dataset, measurement, data_type, x, y, x_vals, y_vals):
    """
    Adds a measurement data point to the given dataset. The dataset should have the structure
    of an xarray DataArray.

    Parameters
    ----------
    dataset : xarray.DataArray
        The dataset to add the measurement data point to.
    measurement : dict
        A dictionary containing the measurement data.
    data_type : str
        The type of measurement, either 'EDX', 'MOKE', or 'XRD'.
    x : float
        The x position of the measurement.
    y : float
        The y position of the measurement.
    x_vals : list
        A list of all the x values in the dataset.
    y_vals : list
        A list of all the y values in the dataset.

    Returns
    -------
    None
    """

    for key in measurement.keys():
        if key not in dataset:
            data = measurement[key]
            # Special case for intensity where it is stored as shape (3000, 2)
            if key == "intensity":
                data = measurement[key][0][:2986]

            dataset[key] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals, data], dims=["y", "x", key]
            )

        if key == "intensity":
            dataset[key].loc[{"y": y, "x": x}] = measurement[key][0][:2986]
        else:
            dataset[key].loc[{"y": y, "x": x}] = measurement[key]

    # if data_type.lower() == "edx":
    #     # Generate a new DataArray the first time a new data type is encountered
    #     if "counts" not in dataset:
    #         dataset["counts"] = xr.DataArray(
    #             np.nan,
    #             coords=[y_vals, x_vals, measurement["energy"]],
    #             dims=["y", "x", "energy"],
    #         )
    #     # Add the measurement data point to the existing DataArray
    #     dataset["counts"].loc[{"y": y, "x": x, "energy": measurement["energy"]}] = (
    #         measurement["counts"]
    #     )

    # if data_type.lower() == "moke":
    # if "Loops" not in dataset:
    #     dataset["Loops"] = xr.DataArray(
    #         np.nan,
    #         coords=[y_vals, x_vals, ["magnetization", "applied field"], n_indexes],
    #         dims=["y", "x", "index_value", "n_indexes"],
    #     )
    # dataset["Loops"].loc[
    #     {"y": y, "x": x, "index_value": "magnetization", "n_indexes": n_indexes}
    # ] = measurement["magnetization"]
    # dataset["Loops"].loc[
    #     {"y": y, "x": x, "index_value": "applied field", "n_indexes": n_indexes}
    # ] = measurement["applied field"]

    # if data_type.lower() == "xrd":
    #     if "counts" not in dataset:
    #         dataset["counts"] = xr.DataArray(
    #             np.nan,
    #             coords=[y_vals, x_vals, measurement["angle"]],
    #             dims=["y", "x", "angle"],
    #         )
    #     dataset["counts"].loc[{"y": y, "x": x, "angle": measurement["angle"]}] = (
    #         measurement["counts"]
    #     )

    return None


def get_current_dataset(data_type, dataset_edx, dataset_moke, dataset_xrd):
    """
    Returns the current dataset based on the given data_type.

    Parameters
    ----------
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD'.
    dataset_edx : xarray.Dataset
        The Dataset containing the EDX data.
    dataset_moke : xarray.Dataset
        The Dataset containing the MOKE data.
    dataset_xrd : xarray.Dataset
        The Dataset containing the XRD data.

    Returns
    -------
    xarray.Dataset
        The current dataset based on the given data_type.
    """

    if data_type.lower() == "edx":
        current_dataset = dataset_edx
    elif data_type.lower() == "moke":
        current_dataset = dataset_moke
    elif data_type.lower() == "xrd":
        current_dataset = dataset_xrd

    return current_dataset


def get_measurement_data(hdf5_file, datatype, exclude_wafer_edges=True):
    """
    Reads measurement data from the given HDF5 file and returns an xarray DataTree object containing the measurement data.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD', or 'all'.
    exclude_wafer_edges : bool, optional
        If True, the function will exclude the data measured at the edges of the wafer from the returned DataTree. Defaults to True.

    Returns
    -------
    xarray.DataTree
        A DataTree object containing the measurement data.
    """

    # Check if data_type is valid
    if datatype.lower() == "all":
        datatypes = ["EDX", "MOKE", "XRD"]

    elif not datatype.lower() in ["edx", "moke", "xrd"]:
        print("data_type must be one of 'EDX', 'MOKE', 'XRD' or 'all'.")
        return 1
    else:
        datatypes = [datatype]

    measurement_tree = xr.DataTree(name="Measurement Data")
    dataset_edx = xr.Dataset()
    dataset_moke = xr.Dataset()
    dataset_xrd = xr.Dataset()

    for data_type in datatypes:
        print("Reading", data_type)
        positions = get_all_positions(hdf5_file, data_type=data_type)
        x_vals = sorted(set([pos[0] for pos in positions]))
        y_vals = sorted(set([pos[1] for pos in positions]))

        # Looking for modified datasets
        with h5py.File(hdf5_file, "r") as h5f:
            group = make_group_path(hdf5_file, data_type)
            if "hdf5_reader" in h5f[group].attrs.keys():
                print("Modified dataset found for", data_type)
                print(h5f[group].attrs["note"])

        # Add measurement data
        for x, y in positions:
            if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
                continue
            measurement, units = search_measurement_data_from_type(
                hdf5_file, data_type, x, y
            )
            current_dataset = get_current_dataset(
                data_type, dataset_edx, dataset_moke, dataset_xrd
            )
            add_measurement_data(
                current_dataset, measurement, data_type, x, y, x_vals, y_vals
            )

        # Add units for x, y positions for all datasets
        position_units = get_position_units(hdf5_file, data_type=data_type)
        current_dataset["x"].attrs["units"] = position_units["x_pos"]
        current_dataset["y"].attrs["units"] = position_units["y_pos"]

        # Add units for scan axis in all datasets
        for key in units.keys():
            if data_type.lower() != "moke":
                if key in current_dataset:
                    current_dataset[key].attrs["units"] = units[key]
            # else:
            #     if (
            #         key in current_dataset["index_value"]
            #         and "units" not in current_dataset["Loops"].attrs
            #     ):
            #         print(units[key])
            #         current_dataset["Loops"].attrs["units"] = units

    # Add datasets to the xarray DataTree
    measurement_tree["EDX"] = dataset_edx
    measurement_tree["MOKE"] = dataset_moke
    measurement_tree["XRD"] = dataset_xrd

    return measurement_tree


# def get_xrd_images(hdf5_file, exclude_wafer_edges=True):
#     dataset = xr.Dataset()

#     positions = _get_all_positions(hdf5_file, data_type="xrd")
#     x_vals = sorted(set([pos[0] for pos in positions]))
#     y_vals = sorted(set([pos[1] for pos in positions]))

#     for x, y, nb_scan in tqdm(positions):
#         if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
#             continue

#         group_path = make_group_path(["XRD", nb_scan, "image"])
#         image = get_xrd_image(hdf5_file, group_path)["2D_Camera_Image"]

#         if "image" not in dataset:
#             dataset["image"] = xr.DataArray(
#                 np.nan,
#                 coords=[
#                     y_vals,
#                     x_vals,
#                     np.arange(image.shape[0]),
#                     np.arange(image.shape[1]),
#                 ],
#                 dims=["y", "x", "pixel x", "pixel y"],
#             )
#         dataset["image"].loc[{"y": y, "x": x}] = image

#     return dataset


def create_simplified_dataset(hdf5_file, hdf5_save_file):
    """
    Creates a simplified HDF5 dataset with the measurement data sorted by x and y position coordinates.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the input HDF5 file.
    hdf5_save_file : str or pathlib.Path
        The path to the output HDF5 file.
    """

    group_list = ["edx", "moke", "xrd"]
    coord_list = [
        "({:.1f},{:.1f})".format(float(x), float(y))
        for x in range(-40, 45, 5)
        for y in range(-40, 45, 5)
    ]

    with h5py.File(hdf5_file, "r") as h5f, h5py.File(hdf5_save_file, "w") as h5f_save:

        for group in h5f["./"]:
            try:
                datatype = h5f[f"{group}"].attrs["HT_type"]
            except KeyError:
                continue

            if datatype in group_list:
                for coord in coord_list:
                    # Check if the group already exists
                    saved_group_coord = f"{group}/{coord}"

                    if coord not in h5f_save:
                        try:
                            instrument = h5f[f"{saved_group_coord}"]["instrument"]
                        except KeyError:
                            continue

                        h5f_save.create_group(f"{coord}")

                        # Create x and y position datasets
                        h5f_save[f"{coord}"].create_dataset(
                            "x_pos", data=instrument["x_pos"]
                        )
                        h5f_save[f"{coord}"].create_dataset(
                            "y_pos", data=instrument["y_pos"]
                        )
                        h5f_save[f"{coord}"]["x_pos"].attrs["units"] = instrument[
                            "x_pos"
                        ].attrs["units"]
                        h5f_save[f"{coord}"]["y_pos"].attrs["units"] = instrument[
                            "y_pos"
                        ].attrs["units"]
                        h5f_save[f"{coord}"]["x_pos"].attrs["HT_type"] = "position"
                        h5f_save[f"{coord}"]["y_pos"].attrs["HT_type"] = "position"

                    if coord not in h5f[f"{group}"].keys():
                        # Giving NaN values for missing data
                        node = h5f_save[f"{coord}"]
                        results = h5f[f"{group}/(0.0,0.0)"]["results"]
                        # If EDX (but should never happened)
                        if datatype == "edx":
                            for key in results.keys():
                                if "Element" in key:
                                    node.create_dataset(key.split(" ")[-1], data=np.nan)
                                    node[key.split(" ")[-1]].attrs["units"] = "at.%"
                                    node[key.split(" ")[-1]].attrs["HT_type"] = datatype
                        # If MOKE
                        elif datatype == "moke":
                            for key in results.keys():
                                if key == "coercivity_m0":
                                    node.create_dataset(key, data=np.nan)
                                    node[key.split(" ")[-1]].attrs[
                                        "units"
                                    ] = "Tesla (T)"
                                    node[key.split(" ")[-1]].attrs["HT_type"] = datatype
                                elif key == "max_kerr_rotation":
                                    pass
                                    """node.create_dataset(key, data=np.nan)
                                    node[key.split(" ")[-1]].attrs["HT_type"] = datatype"""

                        # If XRD
                        elif datatype == "xrd":
                            saving_result_list = ["A", "B", "C", "phase_fraction"]

                            for phase in results["phases"].keys():
                                for saving_key in saving_result_list:
                                    if saving_key in results["phases"][phase].keys():
                                        node.create_dataset(
                                            f"{phase}_{saving_key}", data=np.nan
                                        )
                        continue

                    # Creates new dataset with current datatype
                    if datatype == "edx":
                        node = h5f_save[f"{coord}"]
                        results = h5f[f"{group}/{coord}"]["results"]
                        for key in results.keys():
                            if "Element" in key:
                                try:
                                    node.create_dataset(
                                        key.split(" ")[-1],
                                        data=results[key]["AtomPercent"][()],
                                    )
                                    node[key.split(" ")[-1]].attrs["units"] = results[
                                        key
                                    ]["AtomPercent"].attrs["units"]
                                    node[key.split(" ")[-1]].attrs["HT_type"] = datatype
                                except KeyError:
                                    reference_results = h5f[f"{group}/(0.0,0.0)"][
                                        "results"
                                    ]
                                    if (
                                        key in reference_results.keys()
                                        and "AtomPercent"
                                        in reference_results[key].keys()
                                    ):
                                        node.create_dataset(
                                            key.split(" ")[-1],
                                            data=np.nan,
                                        )
                                        node[key.split(" ")[-1]].attrs["units"] = (
                                            reference_results[key]["AtomPercent"].attrs[
                                                "units"
                                            ]
                                        )
                                        node[key.split(" ")[-1]].attrs[
                                            "HT_type"
                                        ] = datatype

                    elif datatype == "moke":
                        node = h5f_save[f"{coord}"]
                        results = h5f[f"{group}/{coord}"]["results"]
                        for key in results.keys():
                            if key == "coercivity_m0":
                                node.create_dataset(
                                    key,
                                    data=results[key]["mean"][()],
                                )
                                node[key].attrs["units"] = "Tesla (T)"
                                node[key].attrs["HT_type"] = datatype
                            elif key == "max_kerr_rotation":
                                pass
                                """ node.create_dataset(
                                    key,
                                    data=results[key][()],
                                )
                                node[key].attrs["units"] = "Degrees (°)"
                                node[key].attrs["HT_type"] = datatype """

                    elif datatype == "xrd":
                        saving_result_list = ["A", "B", "C", "phase_fraction"]
                        node = h5f_save[f"{coord}"]
                        results = h5f[f"{group}/{coord}"]["results/phases"]
                        measurement = h5f[f"{group}/{coord}"]["measurement"]

                        # Fetching the results
                        for phase in results.keys():
                            for result in saving_result_list:
                                if result in results[phase].keys():

                                    node.create_dataset(
                                        f"{phase}_{result}",
                                        data=(
                                            str(results[phase][result][()])
                                            .strip()
                                            .split("+-")[0]
                                        ),
                                    )
                                    try:
                                        node[f"{phase}_{result}"].attrs["units"] = (
                                            results[phase][result].attrs["units"]
                                        )
                                        node[f"{phase}_{result}"].attrs[
                                            "HT_type"
                                        ] = datatype
                                    except KeyError:
                                        # Taking into account missing attributes
                                        pass

                        # Fetching integrated intensity
                        node.create_dataset(
                            "CdTe_integrate_intensity",
                            data=measurement["CdTe_integrate/intensity"][()],
                        )
                        node["CdTe_integrate_intensity"].attrs[
                            "units"
                        ] = "arbitrary unit (a.u.)"
                        node["CdTe_integrate_intensity"].attrs["HT_type"] = datatype

                        # Fetching integrated q
                        node.create_dataset(
                            "CdTe_integrate_q",
                            data=measurement["CdTe_integrate/q"][()],
                        )
                        node["CdTe_integrate_q"].attrs["units"] = "Angstrom^-1 (A^-1)"
                        node["CdTe_integrate_q"].attrs["HT_type"] = datatype

                        # Fetching CdTe image
                        node.create_dataset(
                            "CdTe",
                            data=measurement["CdTe"][()],
                        )
                        node["CdTe"].attrs["HT_type"] = datatype

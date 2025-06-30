import os
import argparse


def check_readme_exists(folder_path, verbose='on'):
    readme_path = os.path.join(folder_path, 'README')
    if os.path.isfile(readme_path):
        if verbose in ['on', 'debug']:
            print(f"File 'README' exists.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(
                f"File 'README' does not exist in the folder '{folder_path}'.")
        return False


def check_subfolders_exist(folder_path, check_optional_subf=True, verbose='on'):
    mandatory_subfolders = ['GS', 'GS/x', 'GS/z', 'Jij', 'MC']
    non_mandatory_subfolders = ['GS/y']

    mandatory_exist = []
    non_mandatory_exist = []

    if verbose in ['debug']:
        print("# In check_subfolders_exist function.")

    for subfolder in mandatory_subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            if verbose in ['on', 'debug']:
                print(f"Mandatory subfolder '{subfolder}' exists.")
            mandatory_exist.append(True)
        else:
            if verbose in ['on', 'debug']:
                print(
                    f"Mandatory subfolder '{subfolder}' does not exist in the folder '{folder_path}'.")
            mandatory_exist.append(False)
    if check_optional_subf:
        for subfolder in non_mandatory_subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                if verbose in ['on', 'debug']:
                    print(f"Optional subfolder '{subfolder}' exists.")
                non_mandatory_exist.append(True)
            else:
                if verbose in ['on', 'debug']:
                    print(
                        f"Optional subfolder '{subfolder}' does not exist in the folder '{folder_path}'.")
                non_mandatory_exist.append(False)
    else:
        non_mandatory_exist = [None] * len(non_mandatory_subfolders)
        if verbose in ['on', 'debug']:
            print("Optional subfolders are not checked.")
        if verbose in ['debug']:
            print("non_mandatory_exist: ", non_mandatory_exist)

    return all(mandatory_exist), non_mandatory_exist


def check_structure_cif_file_exists(folder_path, verbose='on'):
    structure_file_path = os.path.join(folder_path, 'structure.cif')
    if os.path.isfile(structure_file_path):
        if verbose in ['on', 'debug']:
            print(f"File 'structure.cif' exists.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(f"File 'structure.cif' does not exist.")
        return False


def check_out_last_file_exists(folder_path, verbose='on'):
    out_last_file_path = os.path.join(folder_path, 'out_last')
    if os.path.isfile(out_last_file_path):
        if verbose in ['on', 'debug']:
            print(f"File 'out_last' exists in the folder '{folder_path}'.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(
                f"File 'out_last' does not exist in the folder '{folder_path}'.")
        return False


def check_posfile_file_exists(folder_path, verbose='on'):
    posfile_file_path = os.path.join(folder_path, 'posfile')
    if os.path.isfile(posfile_file_path):
        if verbose in ['on', 'debug']:
            print(
                f"File 'posfile' exists in the folder '{folder_path}'.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(
                f"File 'posfile' does not exist in the folder '{folder_path}'.")
        return False

# TODO: check is jfile needed anyhow
def check_jfile_file_exists(folder_path, verbose='on'):
    jfile_file_path = os.path.join(folder_path, 'jfile')
    if os.path.isfile(jfile_file_path):
        if verbose in ['on', 'debug']:
            print(
                f"File 'jfile' exists in the folder '{folder_path}'.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(
                f"File 'jfile' does not exist in the folder '{folder_path}'.")
        return False


def check_momfile_file_exists(folder_path, verbose='on'):
    momfile_file_path = os.path.join(folder_path, 'momfile')
    if os.path.isfile(momfile_file_path):
        if verbose in ['on', 'debug']:
            print(
                f"File 'momfile' exists in the folder '{folder_path}'.")
        return True
    else:
        if verbose in ['on', 'debug']:
            print(
                f"File 'momfile' does not exist in the folder '{folder_path}'.")
        return False


def check_structure(
    folder_path,
    check_subfolders=True,
    check_optional_subf=True,
    check_README=False,
    check_structure_cif=True,
    check_out_last_files=True,
    verbose='on',
):

    if verbose in ['debug']:
        print("# In check_structure function.")
    main_folder_exists = False
    readme_exists = False
    mandatory_subfolders_exist = False
    optional_subfolders_exist = False
    structure_cif_exists = False
    out_last_x_exists = False
    out_last_y_exists = False
    out_last_z_exists = False

    if not os.path.isdir(folder_path):
        if verbose in ['on', 'debug']:
            print(
                f"The main folder '{folder_path}' for the dataset does not exist.")
        main_folder_exists = False
    else:
        if verbose in ['on', 'debug']:
            print(f"The main folder '{folder_path}' for the dataset exists.")
        main_folder_exists = True

        if check_README:
            readme_exists = check_readme_exists(folder_path, verbose)
        if check_subfolders:
            mandatory_subfolders_exist, optional_subfolders_exist = check_subfolders_exist(
                folder_path, check_optional_subf, verbose)
            if verbose in ['debug']:
                print(
                    f"Mandatory subfolders exist: {mandatory_subfolders_exist}")
                print(
                    f"Optional subfolders exist: {optional_subfolders_exist}")
            if mandatory_subfolders_exist:
                if check_out_last_files:
                    out_last_x_exists = check_out_last_file_exists(
                        folder_path+'/GS/x', verbose)
                    out_last_z_exists = check_out_last_file_exists(
                        folder_path+'/GS/z', verbose)

                posfile_exists = check_posfile_file_exists(
                    folder_path+'/MC', verbose)
                jfile_exists = check_jfile_file_exists(
                    folder_path+'/MC', verbose)
                momfile_exists = check_momfile_file_exists(
                    folder_path+'/MC', verbose)

                if (posfile_exists or momfile_exists):
                    if verbose in ['on', 'debug']:
                        print(
                            f"Files 'posfile', 'jfile', or 'momfile' exist in the folder '{folder_path}/MC'.")
                    mom_pos_file_exists = True
                else:
                    if verbose in ['on', 'debug']:
                        print(
                            f"None of the files 'posfile', 'jfile', or 'momfile' exist in the folder '{folder_path}/MC'.")
                    mom_pos_file_exists = False

            if optional_subfolders_exist and check_optional_subf:
                if check_out_last_files:
                    out_last_y_exists = check_out_last_file_exists(
                        folder_path+'/GS/y', verbose)
        if check_structure_cif:
            structure_cif_exists = check_structure_cif_file_exists(
                folder_path, verbose)

        return_values = [main_folder_exists, readme_exists, mandatory_subfolders_exist,
                         structure_cif_exists, out_last_x_exists, out_last_z_exists,
                         mom_pos_file_exists]

        if verbose == 'debug':
            print(f"All return values: {return_values}")

        if all(return_values):
            if optional_subfolders_exist == [True] * len(optional_subfolders_exist):
                if verbose in ['on', 'debug']:
                    print("## Optional subfolders exist.")
                    if out_last_y_exists:
                        print("## Optional out_last_y file exists.")
            else:
                if optional_subfolders_exist == [False] * len(optional_subfolders_exist):
                    if verbose in ['on', 'debug']:
                        print("## Optional subfolders do not exist.")
                else:
                    if optional_subfolders_exist == [None] * len(optional_subfolders_exist):
                        if verbose in ['on', 'debug']:
                            print("## Optional subfolders are not checked.")
            if verbose in ['off', 'on', 'debug']:
                print("# All checks passed.")
            return True
        else:
            if verbose in ['off', 'on', 'debug']:
                print("# Some checks failed.")
            return False


def main(args):
    dataset_name = 'Co2Fe2H4'  # Default value
    check_optional_subf = True  # Default value
    verbose = 'on'

    if args.dataset_name:
        dataset_name = args.dataset_name

    if args.check_optional_subf is not None:
        check_optional_subf = args.check_optional_subf

    if args.verbose:
        verbose = args.verbose

    check_structure(dataset_name, check_optional_subf=check_optional_subf, check_README=True,
                    check_subfolders=True, check_structure_cif=True, check_out_last_files=True, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check folder structure of a simulation data set for the 'Electronic' and 'Atomistic' model.")
    parser.add_argument('-dataset_name', type=str, default='Co2Fe2H4',
                        help='Name of the folder or dataset to check (default: Co2Fe2H4)')
    parser.add_argument('-check_optional_subf', action='store_true',
                        dest='check_optional_subf', help='Check optional subfolders (default: True)')
    parser.add_argument('-no_check_optional_subf', action='store_false',
                        dest='check_optional_subf', help='Do not check optional subfolders')
    parser.add_argument('-verbose', type=str, choices=[
                        'off', 'on', 'debug'], default='on', help='Set verbosity level (default: on)')

    args = parser.parse_args()
    main(args)

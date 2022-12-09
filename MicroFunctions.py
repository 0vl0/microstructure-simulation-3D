"""
Functions to deal with microstructures
"""
import numpy as np 
from generate_db_micro import get_tab_names
from FileManager import FileManager
import os

def get_concentration_phase(path_npy, number_phases):
    """
    Get concentrations of the different phases inside of microstructure
    Inputs:
        path_npy: path to the npy file of the microstructure 
    """
    micro = np.load(path_npy)
    Nx, Ny, Nz = micro.shape
    cube_length = Nx*Ny*Nz
    tab_concentration = np.array(list(map(lambda i: len(np.where(micro == i+2)[0]), range(number_phases))))/cube_length
    return tab_concentration

def compute_volumetric_fractions_void(id_micro, fM):
    """
    Compute volumetric fractions inside of cylinder encircled by void
    """
    micro = np.load(fM._get_path_npy_image_void(id_micro))
    number_phases = 3
    Nx, Ny, Nz = micro.shape
    tab_concentration = np.array(list(map(lambda i: len(np.where(micro == i+1)[0]), range(number_phases))))
    N_voxel_inside_cylinder = np.sum(tab_concentration)
    tab_concentration = tab_concentration/N_voxel_inside_cylinder
    f_2 = tab_concentration[1]
    f_3 = tab_concentration[2]
    return f_2, f_3
    #print('tab_concentration/N = ', tab_concentration/N_voxel_inside_cylinder)

def save_volumetric_fractions_void(id_micro, fileManager):
    """
    Generate properties file if not done during computation
    """
    f_2, f_3 = compute_volumetric_fractions_void(id_micro, fileManager)
    d = {}
    d["id_micro"] = id_micro
    d["f_2"] = f_2
    d["f_3"] = f_3
    fileManager.save_properties_file(id_micro, d)

def save_volumetric_fractions(id_micro, fileManager):
    """
    Generate properties file if not done during computation
    """
    f_2, f_3 = get_concentration_phase(id_micro, fileManager)
    d = {}
    d["id_micro"] = id_micro
    d["f_2"] = f_2
    d["f_3"] = f_3
    fileManager.save_properties_file(id_micro, d)


if __name__ == "__main__":

    """
    fileManager = FileManager()
    names = os.listdir(fileManager._get_path_microstructures_directory())
    for id_micro in names:
        save_volumetric_fractions_void(id_micro, fileManager)
    """


    """
    tab_names = ["ground_truth_void"]
    for name in tab_names:
        fileManager._create_micro_directory(name)
        tab_concentration = get_concentration_phase(fileManager._get_path_npy_image(name), 2)
        generate_properties_file(name, tab_concentration, fileManager)
    """
    



from scipy.stats import poisson
from FileManager import SAVE_PATH, FileManager
from MixedMicrostructure import MixedMicrostructure
from multiprocessing import Pool
import re
import os
import matplotlib.pyplot as plt
import pandas as pd

# Data from covariance analysis of cross-section real microstructures
ROCK_CONCENTRATION = 0.075
MUD_CONCENTRATION = 0.112
CUBE_SIZE = 100
LOADING_STEP = 1e-5
N_STEP = 50

def load_voxel_concentration():
    with open("/home/vludvig/nas/stage/python/clustering_v1/voxel_concentration.txt", "r") as f:
        average_concentration = float(f.readline())
    return average_concentration

def get_number_inclusion():
    """
    Cube is filled with inclusions of ROCK and MUD types.
    The number of inclusions is chosen to match the real concentrations.
    The average concentration is computed for 50^3 voxel forms.
    As we generate 100³ voxel forms, the average concentration should by divided by 8.
    """
    # average concentration of a 3D form in the 50^3 space
    avg_c = load_voxel_concentration()
    print("avg_c in 50^3 voulme: ", avg_c)
    avg_c *= (50/CUBE_SIZE)**3
    print("avg_c in 100³ volume: ", avg_c)
    N_ROCK = ROCK_CONCENTRATION/avg_c
    N_MUD = MUD_CONCENTRATION/avg_c
    return N_ROCK, N_MUD

def generate_microstructures(N_experiment, cube_size, loading_step, n_step, poisson=False):
    """
    Generate database of microstructures for later mechanical computations.
    Inputs:
        - poisson: if True add random noise to number of inclusions with a poisson distribution
    """
    print("Generating microstructures....")
    n_rock, n_mud = get_number_inclusion()
    if poisson:
        n_rock_tab = poisson.rvs(mu=n_rock, size=N_experiment)
        n_mud_tab = poisson.rvs(mu=n_mud, size=N_experiment)
    else:
        n_rock_tab, n_mud_tab = [round(n_rock)]*N_experiment, [round(n_mud)]*N_experiment
    mixed_micro = MixedMicrostructure(100, (0,0))
    for i in range(301, N_experiment+1):
        #print("i = ", i)
        n_rocks, n_mud = n_rock_tab[i-1], n_mud_tab[i-1]
        #print("n_rocks= {}\tn_mud = {}".format(n_rocks, n_mud))
        mixed_micro.set_number_forms((n_rocks, n_mud))
        mixed_micro.generate_microstructure(get_name_microstructure(i, cube_size, loading_step, n_step, n_rocks, n_mud))

def get_name_microstructure(i, cube_size, loading_step, n_step, n_rocks, n_mud):
    """
    Get name of microstructure corresponding to specified parameters
    """
    return "_".join((str(i), str(cube_size), str(loading_step), str(n_step), str(n_rocks), str(n_mud)))

def write_id_micro(path_file_names, fileManager):
    """
    Write identifiers of generated microstructure in a file.
    Useful to divide computations
    """
    names = os.listdir(fileManager._get_path_microstructures_directory())
    regexp  = r"[0-9]{1,4}_100_1e-05_50_[0-9]{1,2}_[0-9]{1,2}"
    print("regexp = ", regexp)
    m = re.findall(regexp, " ".join(names))
    with open(path_file_names, "w") as f:
        f.writelines("\n".join(m))

def generate_mechanical_results(fileManager, id_micro):
    """
    Generate all mechanical results for micro identified in micro_names tab
    """
    for id in id_micro:
        print("generating mechanical results, id_micro = {}".format(id))
        fileManager.generate_results(id_micro=id, N_STEP=N_STEP, loading_step=LOADING_STEP) 

def generate_elasticity_results(fileManager, id_micro_tab):
    """
    Generate all elasticity results for microstructures identified in id_micro_tab.
    """
    
    for id in id_micro_tab:
        fileManager._generate_all_elasticity_results(id_micro=id, loading_step=LOADING_STEP, N_STEP=2)

def get_tab_names(file_names):
    """
    Get names of microstructures written in file_names.
    Each line of the txt file contains one name.
    """
    with open(file_names, "r") as f:
        lines = f.readlines()
        tab = [line.rstrip() for line in lines]
    return tab

def save_sorted_tab_names(path_file_name, fileManager):
    """
    Sort ids according to last sigma value.
    """
    tab_names = get_tab_names(path_file_name)
    tab_names_sorted = sorted(tab_names, key=fileManager._get_last_sigma)
    with open(path_file_name.split(".txt")[0] + "_sorted.txt", "w") as f:
        f.writelines("\n".join(tab_names_sorted))

def concat_properties(ids_micro, fileManager):
    """
    Concat csv properties files
    """
    tab_ids = get_tab_names(ids_micro)
    df_tab = [pd.read_csv(fileManager._get_path_properties_file(id_micro)).set_index("id_micro") for id_micro in tab_ids]
    return pd.concat(df_tab)

def write_concac_properties(pd_concat, fileManager):
    """Write concatenated properties to global_results folder"""
    fileManager.save_global_properties_file(pd_concat)

def generate_global_properties(ids_micro_name, fileManager):
    """
    Generate and save global properties file
    """
    pd_concat = concat_properties(ids_micro_name, fileManager)
    print("pd_concat = ", pd_concat)
    write_concac_properties(pd_concat, fileManager)

def complete_properties_file(ids_micro, fileManager):
    """
    Add last_sigma and elasticity matrix to properties
    """
    for id_micro in ids_micro:
        fileManager._add_all_properties(id_micro)

if __name__ == "__main__":
    fileManager = FileManager()
    #write_id_micro("/home/vludvig/nas/stage/python/clustering_v1/micro_ids3000.txt", fileManager)
    #generate_microstructures(N_experiment=3000, cube_size=CUBE_SIZE, loading_step=LOADING_STEP, n_step=N_STEP, poisson=False)
    #fileManager._create_global_directory()
    #generate_global_properties("micro_ids2.txt", fileManager)
    #micro_ids = "/home/vludvig/nas/stage/python/clustering_v1/micro_ids2.txt"
    #save_sorted_tab_names("/home/vludvig/nas/stage/python/clustering_v1/micro_ids2.txt" ,fileManager)
    #plt.show()
    
    
    tab_names = get_tab_names("micro_ids3000.txt")
    
    for id_micro in tab_names[1528:2000]:
        fileManager._generate_all_elasticity_results(id_micro, LOADING_STEP, 2)
    
    #generate_global_properties(tab_names, fileManager)
    #concat_properties(tab_names)
    #plt.figure(figsize=(9.6,7.2))
    #fileManager.plot_db_micro(tab_names[:300], normalized=True, positive=True)
    #plt.savefig("/home/vludvig/nas/stage/python/clustering_v1/experiments/experiment2/images/loading_curve_e2_normalized.png")
    #plt.show()
    #plt.savefig("/home/vludvig/nas/stage/python/clustering_v1/experiments/experiment2/images/loading_curve_e2.png")
    
    
    #generate_mechanical_results(fileManager, tab_names[1171:1200])
    
    
    #generate_mechanical_results(fileManager, tab_names[1171:2100])
    #generate_mechanical_results(fileManager, tab_names[2884:])
    #N_rock, N_mud = get_number_inclusion()
    #print("n_rock = {}, n_mud = {}".format(N_rock, N_mud))
__Get volumetric fractions of each phase__ <br>
See get_concentration_phase() from [MicroFunctions.py](). The microstructures should have been created beforehand.

__Get volumetric fractions when void has been added__ <br>
For some microstructures void is added on the outside to have more realistic boundary conditions. The computation of the volumetric fractions needs to take into account this void. See compute_volumetric_fractions_void from [MicroFunctions.py]().

__Save volumetric fractions__ <br>
See save_volumetric_fractions and save_volumetric_fractions_void from [MicroFunctions.py]() to compute and save volumetric fractions of a newly-generated microstructure.

__Generate id_micro txt file__ <br>
Some functions require a txt file with all the id of the microstructures that you wish to process. The microstructures should have been generated when you generate the txt file. To generate the txt file, see the function write_ids_micro from [generate_db_micro.py](generate_db_micro.py)

__Sort the microstructures based on the elasticity threshold__ <br>
The id_micro txt file must be generated before. <br> Then see save_sorted_tab_names from [generate_db_micro.py]()

__Concatenate the properties files__ <br>
Once the properties have been computed and saved for each of the microstructures of the experiment, you can concatenate all these properties file into a big global properties file. See generate_global_properties from [generate_db_micro.py]()

__Plot loading curve__ <br>
To plot the loading curve, see plot_from_csv() from [FileManager.py]().
Sigma and epsilon values should have been computed and saved beforehand. You need to call plt.plot() after calling the function.
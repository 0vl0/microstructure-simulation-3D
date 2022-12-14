import os, subprocess
import numpy.random as rdn
import numpy as np, csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from multiprocessing import Pool
import pandas as pd, time, matplotlib.image
from sklearn import linear_model
SAVE_PATH = '/home/vl/stage_cmm/experiment'
PATH_MORPHHOM = '/home/vludvig/morphhom_taurus_v2'
IMAGE_PATH = '/home/vl/stage_cmm/experiment/microstructures/'
LIST_DEL_FILES = ['_Plas', '_DEpsilon_xx', '_DEpsilon_xy', '_DEpsilon_xz', '_DEpsilon_yy', '_DEpsilon_yz', '_DEpsilon_zz', '_DSigma_xx', '_DSigma_xy', '_DSigma_xz', '_DSigma_yy', '_DSigma_yz', '_DSigma_zz', '_DPlas', '_c']
TENSORS = ['_Epsilon_xx', '_Epsilon_xy', '_Epsilon_xz', '_Epsilon_yy', '_Epsilon_yz', '_Epsilon_zz', '_Sigma_xx', '_Sigma_xy', '_Sigma_xz', '_Sigma_yy', '_Sigma_yz', '_Sigma_zz']
LIST_MOVE_FILES = TENSORS
LIST_MOVE_FILES_END = ['M']
DIRECTIONS = {'xx':'1 0 0 0 0 0',  'xy':'0 .5 0 0 0 0',  'xz':'0 0 .5 0 0 0',  'yy':'0 0 0 1 0 0',  'yz':'0 0 0 0 .5 0',  'zz':'0 0 0 0 0 1'}
D_VOIGT = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

class FileManager:

    def __init__(self):
        self._number_images = self._get_number_images()

    def save_properties_file(self, id_micro, volumic_fraction_dict):
        """
        Save dictionary containing volumic fraction into panda file.
        Used when data has been generated without saving volumic fractions.
        """
        df = pd.DataFrame(data=volumic_fraction_dict, index=[id_micro])
        df.to_csv((self._get_path_properties_file(id_micro)), index=False)

    def _get_number_images(self):
        """
        Get the number of images in the directory of images
        """
        return int(os.popen('cd {} ; ls | wc -l'.format(IMAGE_PATH)).read())

    def _get_name_file(self, id_micro):
        """
        Return name of files generated for microstructure identified by id_micro
        """
        return str(id_micro) + '_'

    def _get_list_del_files_end(self, N_STEP):
        return [
         'perf', '{}_Ux'.format(N_STEP), '{}_Uy'.format(N_STEP), '{}_Uz'.format(N_STEP)]

    def _get_training_image_path(self, id):
        """
        Returns image path of image identified by id
        Inputs:
            IMAGE_DIRECTORY: String, directory containing all the images
            id: int, id of the image
        Output: 
            String, image path
        """
        return IMAGE_PATH + '/mesh_fusion_{}.npy'.format(id)

    def get_random_image(self):
        """
        Return random image from the image folder.
        """
        id_image = rdn.randint(self._number_images)
        image_path = self._get_training_image_path(id_image)
        return np.load(image_path)

    def _get_voxel_concentration(self, id_image):
        """
        Get number of voxel form divided by total space
        """
        path = self._get_training_image_path(id_image)
        img = np.load(path)
        voxel_number = len(np.where(img == 1)[0])
        return voxel_number / (img.shape[0] * img.shape[1] * img.shape[2])

    def _get_average_voxel_concentration(self):
        N_images = self._get_number_images()
        with Pool(16) as (p):
            average_concentration = sum(list(p.map(self._get_voxel_concentration, range(0, N_images)))) / N_images
        return average_concentration

    def save_average_voxel_concentration(self):
        average_voxel_concentration = self._get_average_voxel_concentration()
        with open('voxel_concentration.txt', 'w') as (f):
            f.write(str(average_voxel_concentration))

    def _get_path_raw(self, id_micro):
        """
        Get the path to the raw folder of image identified by id
        """
        return self._get_path_micro(id_micro) + '/raw'

    def _get_path_raw_image(self, id_micro):
        """
        Get path to the raw file of image identified by id
        """
        return self._get_path_raw(id_micro) + '/micro{}.raw'.format(id_micro)

    def _get_path_npy(self, id_micro):
        """
        Get path to npy image file of microstructure identified by id
        """
        return self._get_path_micro(id_micro) + '/npy'

    def _convert_npy_into_raw(self, id_micro):
        """
        Convert npy file into raw using Fortran index order.
        Fortran odrer is used because morphhom is coded in Fortran.
        """
        img_path = self._get_path_npy_image(id_micro)
        img = np.load(img_path)
        img = img.astype(int)
        img_raw_fortran = img.tobytes(order='F')
        print('writing image: ', self._get_path_raw_image(id_micro))
        with open(self._get_path_raw_image(id_micro), 'wb') as (f):
            f.write(img_raw_fortran)

    def _get_raw_from_npy(self, npy_image):
        """
        Similar to convert_npy_into_raw, but returns raw mage instead of saving
        """
        img = npy_image.astype(int)
        return img.tobytes(order='F')

    def _convert_raw_into_npy(self, micro_shape, path_raw):
        """
        Convert raw file into npy, e.g. raw epsilon field
        Inputs:
            micro_shape: shape of the npy microstructure
            raw_img = raw file of the 3D image
        Output:
            3D numpy array reprensenting the raw image
        """
        epsilon_xx = np.fromfile(path_raw)
        return np.reshape(epsilon_xx, micro_shape, order='F')

    def _get_path_npy_image(self, id_micro):
        """
        Get path to the raw file of image identified by id
        """
        return self._get_path_npy(id_micro) + '/micro{}.npy'.format(id_micro)

    def _get_path_npy_image_void(self, id_micro):
        """
        Get path to the npy clyinder image with void  
        """
        return self._get_path_npy(id_micro) + '/{}_void.npy'.format(id_micro)

    def _get_path_raw_image_void(self, id_micro):
        """
        Get path to the raw image with void
        """
        return self._get_path_raw(id_micro) + '/{}_void.raw'.format(id_micro)

    def _get_path_microstructures_directory(self):
        return SAVE_PATH + '/microstructures'

    def _get_path_micro(self, id_micro):
        """
        Get path to folder of micro identified by id_micro
        """
        return self._get_path_microstructures_directory() + '/{}'.format(id_micro)

    def _get_path_res(self, id_micro):
        """
        Get results directory of image identified by id
        """
        return self._get_path_micro(id_micro) + '/results'.format(id_micro)

    def _get_path_tensors(self, id_micro):
        """
        Get path to output tensors
        """
        return self._get_path_res(id_micro) + '/tensors'

    def _get_path_csv(self, id_micro):
        """
        Get path to output csv files
        """
        return self._get_path_res(id_micro) + '/csv'

    def save_npy(self, img, id_micro):
        """
        Save image as npy file
        """
        np.save(self._get_path_npy_image(id_micro), img)

    def generate_results(self, id_micro, N_STEP, loading_step, void=True):
        """
        Generate results the image identified by id, and save them to the corresponding results folder.
        """
        self._create_micro_directory(id_micro)
        micro_npy = np.load(self._get_path_npy_image(id_micro))
        Nx, Ny, Nz = micro_npy.shape
        os.chdir(PATH_MORPHHOM)
        print('loading_step =', loading_step)
        if not void:
            path_raw = self._get_path_raw_image(id_micro)
        else:
            path_raw = self._get_path_raw_image_void(id_micro)
        computation_command = './morphhom -sz {}x{}x{} -mfmt longlong -rm {} -e -10 0 0 0 0 0 -ds -b 3dt mypropGPa -bth 17.94e-3 -bth2 30e-3 -bth3 0.02e-3 -mohr-coulomb3 -34 -34 -40 -40 -34 -34 -disc-rot -cl 1e-5 -ms 5 -si -refn .2 50 -dsu {} {} -t 16 -o {}'.format(Nx, Ny, Nz, path_raw, loading_step, N_STEP, self._get_name_file(id_micro))
        print('computation command : ', computation_command)
        os.system(computation_command)
        self._generate_loading_curve(id_micro, (micro_npy.shape), N_STEP, path_csv=(self._get_path_csv_file(id_micro)), path_loading_curve=(self._get_path_loading_curve(id_micro)), direction='xx')
        self._del_tensors_end(id_micro, N_STEP)
        self._del_files(id_micro, N_STEP)
        self._compress_tensors(id_micro, N_STEP)
        self._move_results(id_micro, N_STEP)

    def _generate_results_elasticity(self, id_micro, loading_step, N_STEP, direction_loading):
        """
        Generate the six elasticity along one loading direction_loading
        Inputs:
            - id_micro: string, identifier of the microstructure (used to locate the corresponding folder)
            - loading_step: float, loading step used to compute the mechanical result
            - N_STEP: int, number of steps of the process
            - direction_loading: string, loading direction_loading, in [xx, yy, zz, yz, zx, xy] 
        """
        self._create_micro_directory(id_micro)
        micro_npy = np.load(self._get_path_npy_image(id_micro))
        Nx, Ny, Nz = micro_npy.shape
        os.chdir(PATH_MORPHHOM)
        if not os.path.isdir(id_micro):
            print('making directory!')
            os.system('mkdir {}'.format(id_micro))
        id_micro_direction = id_micro + '_' + direction_loading
        micro_path = './{}/{}'.format(id_micro, id_micro_direction)
        d = DIRECTIONS[direction_loading]
        computation_command = './morphhom -sz {}x{}x{} -mfmt longlong -rm {} -e {} -ds -b 3dt mypropGPa -bth 14.47e-3 -bth2 30e-3 -bth3 0.02e-3 -mohr-coulomb3 -34 -34 -40 -40 -34 -34 -disc-rot -cl 1e-5 -si -refn .2 50 -dsu {} {} -t 16 -o {}_'.format(Nx, Ny, Nz, self._get_path_raw_image(id_micro), d, self._get_loading_step(loading_direction=direction_loading, loading_step=loading_step), N_STEP, micro_path)
        print('direction loading = ', direction_loading)
        print('computation_command = ', computation_command)
        os.system(computation_command)
        print('Generating loading curve...')
        list(map(lambda d: self._generate_loading_curve(micro_path, (micro_npy.shape), N_STEP, self._get_path_elas_file(id_micro, d_loading=direction_loading, d_obs=d), path_loading_curve=self._get_path_loading_curve_elas(id_micro=id_micro, direction_loading=direction_loading, direction_observed=d), direction=d), D_VOIGT))

    def _get_loading_step(self, loading_direction, loading_step):
        """
        Get loading corresponding to loading_direction.
        Loading is divided by 2 for loading direction xy, xz, yz to generate the right elasticity matrix,
        under Voigt notation.
        """
        if loading_direction in ('xx', 'yy', 'zz'):
            return loading_step
        if loading_direction in ('xy', 'xz', 'yz'):
            return loading_step / 2
        return 0

    def _generate_all_elasticity_results(self, id_micro, loading_step, N_STEP):
        """Generate all elasticity results, i.e. along all directions xx, yy, zz, yz, zx, xy
        Then the elasticity matrix is written with the results"""
        list(map(lambda d: self._generate_results_elasticity(id_micro, loading_step, N_STEP, direction_loading=d), D_VOIGT))
        self._write_elasticity_matrix(id_micro)
        self._save_elasticity_matrix(id_micro)
        self._del_elasticity_folder(id_micro)

    def _del_files(self, id_micro, N_STEP):
        """
        Delete useless files generated in the morphhom folder to save space
        """
        S = 'rm '
        for j in range(1, N_STEP):
            list_files = list(map(lambda s: self._get_name_file(id_micro) + self._convert_to_morphhom_format(j, N_STEP) + s, LIST_DEL_FILES))
            S += ' '.join(list_files)
            S += ' '
        else:
            list_files = list(map(lambda s: self._get_name_file(id_micro) + s, self._get_list_del_files_end(N_STEP)))
            S += ' '.join(list_files)
            os.system(S)

    def _del_tensors_end(self, id_micro, N_STEP):
        """
        Delete all tensors except the ones from the last iteration. 
        Even when compressed, the tensor folder has a size up to 140Gb,
        so tensors are deleted after calculations.
        """
        os.chdir(PATH_MORPHHOM)
        S = 'rm '
        for j in range(1, N_STEP):
            list_files = list(map(lambda s: self._get_name_file(id_micro) + self._convert_to_morphhom_format(j, N_STEP) + s, TENSORS))
            S += ' '.join(list_files)
            S += ' '
        else:
            os.system(S)

    def _create_micro_directory(self, id_micro):
        """
        Make result directory for image identified by id_micro
        """
        self._create_global_directory()
        if not os.path.isdir(self._get_path_micro(id_micro)):
            os.system('mkdir {}'.format(self._get_path_micro(id_micro)))
        if not os.path.isdir(self._get_path_raw(id_micro)):
            os.system('mkdir {}'.format(self._get_path_raw(id_micro)))
        if not os.path.isdir(self._get_path_npy(id_micro)):
            os.system('mkdir {}'.format(self._get_path_npy(id_micro)))
        if not os.path.isdir(self._get_path_properties(id_micro)):
            os.system('mkdir {}'.format(self._get_path_properties(id_micro)))
        if not os.path.isdir(self._get_path_res(id_micro)):
            os.system('mkdir {}'.format(self._get_path_res(id_micro)))
        if not os.path.isdir(self._get_path_csv(id_micro)):
            os.system('mkdir {}'.format(self._get_path_csv(id_micro)))
        if not os.path.isdir(self._get_path_image(id_micro)):
            os.system('mkdir {}'.format(self._get_path_image(id_micro)))
        if not os.path.isdir(self._get_path_ppm(id_micro)):
            os.system('mkdir {}'.format(self._get_path_ppm(id_micro)))
        if not os.path.isdir(self._get_path_3D_lines(id_micro)):
            os.system('mkdir {}'.format(self._get_path_3D_lines(id_micro)))
        if not os.path.isdir(self._get_path_loading(id_micro)):
            os.system('mkdir {}'.format(self._get_path_loading(id_micro)))
        if not os.path.isdir(self._get_path_tensors(id_micro)):
            os.system('mkdir {}'.format(self._get_path_tensors(id_micro)))
        if not os.path.isdir(self._get_path_cs_png(id_micro)):
            os.system('mkdir {}'.format(self._get_path_cs_png(id_micro)))
        self._create_elasticity_folder(id_micro)

    def _get_path_global_results(self):
        """
        Returns path to global results folder
        """
        return SAVE_PATH + '/global_results'

    def _get_path_global_csv(self):
        """
        Get path to global csv folder.
        """
        return self._get_path_global_results() + '/csv'

    def _get_path_global_csv_file(self):
        """
        Get path to properties csv file which contain the properties of all 
        the microstructures.
        """
        return self._get_path_global_csv() + '/global_properties.csv'

    def _create_global_directory(self):
        """
        Create global directory tree
        """
        if not os.path.isdir(SAVE_PATH):
            os.system('mkdir {}'.format(SAVE_PATH))
        if not os.path.isdir(self._get_path_microstructures_directory()):
            os.system('mkdir {}'.format(self._get_path_microstructures_directory()))
        if not os.path.isdir(self._get_path_global_results()):
            os.system('mkdir {}'.format(self._get_path_global_results()))
        if not os.path.isdir(self._get_path_global_csv()):
            os.system('mkdir {}'.format(self._get_path_global_csv()))
        if not os.path.isdir(self.get_path_regression()):
            os.system('mkdir {}'.format(self.get_path_regression()))
        if not os.path.isdir(self._get_path_SVR()):
            os.system('mkdir {}'.format(self._get_path_SVR()))

    def get_path_regression(self):
        """
        Get path to regression folder
        """
        return self._get_path_global_results() + '/regression'

    def _get_path_SVR(self):
        """
        Get path to regression SVR folder
        """
        return self.get_path_regression() + '/SVR'

    def _create_elasticity_folder(self, id_micro):
        if not os.path.isdir(self._get_path_elasticity(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elasticity(id_micro)))
        if not os.path.isdir(self._get_path_elas_xx(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elas_xx(id_micro)))
        if not os.path.isdir(self._get_path_elas_yy(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elas_yy(id_micro)))
        if not os.path.isdir(self._get_path_elas_zz(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elas_zz(id_micro)))
        if not os.path.isdir(self._get_path_elas_xy(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elas_xy(id_micro)))
        if not os.path.isdir(self._get_path_elas_xz(id_micro)): 
            os.system('mkdir {}'.format(self._get_path_elas_xz(id_micro)))
        if not os.path.isdir(self._get_path_elas_yz(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elas_yz(id_micro)))
        if not os.path.isdir(self._get_path_elasticity_matrix(id_micro)):
            os.system('mkdir {}'.format(self._get_path_elasticity_matrix(id_micro)))

    def _get_path_elasticity_matrix(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/elasticity_matrix'

    def _get_path_elasticity_matrix_txt(self, id_micro):
        return self._get_path_elasticity_matrix(id_micro) + '/elasticity_matrix.txt'

    def _get_path_elasticity(self, id_micro):
        return self._get_path_res(id_micro) + '/elasticity'

    def _get_path_elas_xx(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/xx'

    def _get_path_elas_yy(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/yy'

    def _get_path_elas_zz(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/zz'

    def _get_path_elas_xy(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/xy'

    def _get_path_elas_xz(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/xz'

    def _get_path_elas_yz(self, id_micro):
        return self._get_path_elasticity(id_micro) + '/yz'

    def _get_path_elas_xx_file(self, id_micro, direction_observed):
        return self._get_path_elas_xx(id_micro) + '/sigma_epsilon_xx_{}.csv'.format(direction_observed)

    def _get_path_elas_yy_file(self, id_micro, direction_observed):
        return self._get_path_elas_yy(id_micro) + '/sigma_epsilon_yy_{}.csv'.format(direction_observed)

    def _get_path_elas_zz_file(self, id_micro, direction_observed):
        return self._get_path_elas_zz(id_micro) + '/sigma_epsilon_zz_{}.csv'.format(direction_observed)

    def _get_path_elas_xy_file(self, id_micro, direction_observed):
        return self._get_path_elas_xy(id_micro) + '/sigma_epsilon_xy_{}.csv'.format(direction_observed)

    def _get_path_elas_xz_file(self, id_micro, direction_observed):
        return self._get_path_elas_xz(id_micro) + '/sigma_epsilon_xz_{}.csv'.format(direction_observed)

    def _get_path_elas_yz_file(self, id_micro, direction_observed):
        return self._get_path_elas_yz(id_micro) + '/sigma_epsilon_yz_{}.csv'.format(direction_observed)

    def _get_path_elas_file(self, id_micro, d_loading, d_obs):
        """
        Returns path to elasticity csv file depending on value of loading direction d.
        """
        if d_loading == 'xx':
            return self._get_path_elas_xx_file(id_micro, d_obs)
        if d_loading == 'yy':
            return self._get_path_elas_yy_file(id_micro, d_obs)
        if d_loading == 'zz':
            return self._get_path_elas_zz_file(id_micro, d_obs)
        if d_loading == 'xy':
            return self._get_path_elas_xy_file(id_micro, d_obs)
        if d_loading == 'xz':
            return self._get_path_elas_xz_file(id_micro, d_obs)
        if d_loading == 'yz':
            return self._get_path_elas_yz_file(id_micro, d_obs)

    def _get_path_elas(self, id_micro, d):
        """
        Returns path to elasticity csv file depending on value of loading direction d.
        """
        if d == 'xx':
            return self._get_path_elas_xx(id_micro)
        if d == 'yy':
            return self._get_path_elas_yy(id_micro)
        if d == 'zz':
            return self._get_path_elas_zz(id_micro)
        if d == 'xy':
            return self._get_path_elas_xy(id_micro)
        if d == 'xz':
            return self._get_path_elas_xz(id_micro)
        if d == 'yz':
            return self._get_path_elas_yz(id_micro)

    def _get_path_properties(self, id_micro):
        """
        Get path to txt folder
        """
        return self._get_path_micro(id_micro) + '/properties'

    def _get_path_properties_file(self, id_micro):
        """
        Get path to csv volumic fractions file
        """
        return self._get_path_properties(id_micro) + '/{}_properties.csv'.format(id_micro)

    def save_properties_file(self, id_micro, volumic_fraction_dict):
        """
        Save dictionary containing volumic fraction into panda file.
        """
        df = pd.DataFrame(data=volumic_fraction_dict, index=[id_micro])
        df.to_csv((self._get_path_properties_file(id_micro)), index=False)

    def _get_path_loading(self, id_micro):
        """
        Get path to loading curves
        """
        return self._get_path_image(id_micro) + '/loading'

    def _get_path_image(self, id_micro):
        """
        Return path to image folder
        """
        return self._get_path_res(id_micro) + '/images'

    def _get_path_ppm(self, id_micro):
        """
        Get path to ppm image folder (output of morphhom)
        """
        return self._get_path_image(id_micro) + '/ppm'

    def _get_path_ppm_image(self, id_micro, id_image, micro_shape):
        """
        Get path to ppm image, of microstructure number id_micro and image number id_image
        """
        return self._get_path_ppm(id_micro) + '/{}out_{}.ppm'.format(self._get_name_file(id_micro), self._convert_to_morphhom_format(id_image, micro_shape[2]))

    def _generate_images(self, id_micro, micro_shape, N_STEP):
        """
        Generate 2D cross section images, before results are moved.
        It generates a 3D RGB field which represents the norm of the equivalent tensor.
        """
        os.system('./mstats -s {} {} {} -sl {} -eqe {}{}_Epsilon_??'.format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_image_name(id_micro), self._get_name_file(id_micro), N_STEP))

    def _generate_images_zz(self, id_micro, micro_shape, N_STEP):
        """
        Generate 2D cross section images, before results are moved.
        It generates a 3D RGB field which represents the norm of the xx value of the epsilon tensor.
        """
        print('command = {}'.format('./mstats -s {} {} {} -sl {} {}{}_Epsilon_zz'.format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_image_name(id_micro), self._get_name_file(id_micro), N_STEP)))
        os.system('./mstats -s {} {} {} -sl {} {}{}_Epsilon_zz'.format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_image_name(id_micro), self._get_name_file(id_micro), N_STEP))

    def _generate_images_xx(self, id_micro, micro_shape, N_STEP):
        """
        Generate 2D cross section images, before results are moved.
        It generates a 3D RGB field which represents the norm of the xx value of the epsilon tensor.
        """
        print('command = {}'.format('./mstats -s {} {} {} -sl {} {}{}_Epsilon_xx'.format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_image_name(id_micro), self._get_name_file(id_micro), N_STEP)))
        os.system('./mstats -s {} {} {} -sl {} {}{}_Epsilon_xx'.format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_image_name(id_micro), self._get_name_file(id_micro), N_STEP))

    def _generate_images_end(self, id_micro, N_STEP):
        """
        Generate 2D images of the xx compnonent of the deformation field.
        This function should be called once the tensor computations are done, and tensors are
        compressed inside the tensor folder.
        """
        os.chdir(self._get_path_tensors(id_micro))
        (self._get_image_name(id_micro), self._get_name_file(id_micro))
        cmd_unzip = 'gzip -dk {}{}_Epsilon_xx.gz'.format(self._get_name_file(id_micro), N_STEP)
        os.system(cmd_unzip)
        path_epsilon = '{}{}_Epsilon_xx'.format(self._get_name_file(id_micro), N_STEP)
        epsilon_xx = np.fromfile(path_epsilon)
        epsilon_xx = abs(epsilon_xx)
        epsilon_xx = epsilon_xx / max(epsilon_xx)
        epsilon_xx[epsilon_xx < epsilon_xx.mean()] = 0
        mean = epsilon_xx.mean()
        epsilon_xx[epsilon_xx < mean] = epsilon_xx[(epsilon_xx < mean)] * (0.5 * mean)
        epsilon_xx[np.logical_and(mean <= epsilon_xx, epsilon_xx < 0.5)] = 0.5
        print('epsilon_xx.mean() = ', epsilon_xx.mean())
        print('epsilon_xx.max() = ', epsilon_xx.max())
        micro = np.load(self._get_path_npy_image(id_micro))
        epsilon_reshaped_grey = np.reshape(epsilon_xx, (micro.shape), order='F')
        epsilon_rgba = self.grey2RGBA(epsilon_reshaped_grey)
        np.save(self._get_path_3D_lines_image(id_micro), epsilon_rgba)

    def grey2RGBA(self, red_values):
        """
        Convert grey to RGBA with linear function
        """
        green_values = np.zeros_like(red_values)
        blue_values = np.zeros_like(red_values)
        alpha_values = np.ones_like(red_values)
        alpha_values[red_values > 0] = 0
        return np.stack((red_values, green_values, blue_values, alpha_values), axis=3)

    def _get_image_name(self, id_micro):
        """
        Return prefix image name corresponding to identifier id_micro
        """
        return '{}_out'.format(id_micro)

    def _move_images(self, id_micro, micro_shape):
        """
        Move 2D cross section images to result folder.
        """
        image_name = self._get_image_name(id_micro)
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                for i in range(1, micro_shape[2] + 1):
                    os.system('mv ' + PATH_MORPHHOM + '/{}_{}.ppm '.format(image_name, self._convert_to_morphhom_format(i, micro_shape[2])) + self._get_path_ppm(id_micro))
                else:
                    done = True

    def _convert_to_morphhom_format(self, n, Nz):
        """
        Convert integer n to morphhom number format for file numbering.
        """
        s = ''
        if Nz >= 1000:
            s += str(n // 1000)
            n = n - 1000 * (n // 1000)
        if Nz >= 100:
            s += str(n // 100)
            n = n - 100 * (n // 100)
        if Nz >= 10:
            s += str(n // 10)
            n = n - 10 * (n // 10)
        s += str(n)
        return s

    def _move_results(self, id_micro, N_STEP):
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                S = 'mv '
                S += ' '.join(list(map(lambda s: s + '.gz', self._get_tensors_end(id_micro, N_STEP))))
                S += ' ' + self._get_path_tensors(id_micro)
                os.system(S)
                done = True

    def _compress_single_file(self, file_name):
        os.system('gzip ' + file_name)

    def _compress_tensors(self, id_micro, N_STEP):
        print('Compressing tensors...')
        os.chdir(PATH_MORPHHOM)
        with Pool(16) as (p):
            p.map(self._compress_single_file, self._get_tensors_end(id_micro, N_STEP))

    def _get_tensors_end(self, id_micro, N_STEP):
        """
        Returns list of tensors to compress and move at the end
        """
        l = list(map(lambda s: self._get_name_file(id_micro) + str(N_STEP) + s, TENSORS + ['_Plas', '_c']))
        l.extend(list(map(lambda s: self._get_name_file(id_micro) + s, LIST_MOVE_FILES_END)))
        return l

    def _generate_loading_curve(self, id_micro, micro_shape, N_STEP, path_csv, path_loading_curve, direction):
        """
        Fetch data to generate the compression curve, after data has been moved.
        The 11 sigma scalar are written into a csv file, for later processing.
        Inputs:
            id_micro: int, identifier of the microstructure
            direction: string, direction observed
        """
        os.chdir(PATH_MORPHHOM)
        sigma_tab, epsilon_tab = [0] * (N_STEP + 1), [0] * (N_STEP + 1)
        for i in range(1, N_STEP + 1):
            sigma_cmd = self._fetch_sigma_cmd(i, id_micro, micro_shape, N_STEP, direction)
            epsilon_cmd = self._fetch_epsilon_cmd(i, id_micro, micro_shape, N_STEP, direction)
            results_sigma = subprocess.run(sigma_cmd, shell=True, stdout=(subprocess.PIPE))
            results_epsilon = subprocess.run(epsilon_cmd, shell=True, stdout=(subprocess.PIPE))
            sigma_i_tab = results_sigma.stdout.decode('utf-8')
            epsilon_i_tab = results_epsilon.stdout.decode('utf-8')
            #print('sigma_i_tab = ', sigma_i_tab)
            if i % 10 == 0:
                print('sigma_{}_tab = '.format(i), sigma_i_tab)
            sigma_i_float = float(sigma_i_tab.split()[(-1)])
            epsilon_i_float = float(epsilon_i_tab.split()[(-1)])
            sigma_tab[i] = sigma_i_float
            epsilon_tab[i] = epsilon_i_float
        else:
            done = False
            while not done:
                if self._is_connection_broken():
                    time.sleep(5)
                else:
                    self._plot_loading_curve(epsilon_tab, sigma_tab, id_micro, path_loading_curve)
                    self._write_sigma(path_csv, epsilon_tab, sigma_tab)
                    done = True

    def _is_connection_broken(self):
        """
        Return if the connection is broken.
        """
        is_broken = not os.path.isdir(SAVE_PATH)
        if is_broken:
            print('Connection broken, going to sleep for 5s...')
        return is_broken

    def _write_sigma(self, path, epsilon_tab, sigma_tab):
        with open(path, 'w') as (f):
            writer = csv.writer(f)
            writer.writerow(epsilon_tab)
            writer.writerow(sigma_tab)

    def _get_path_csv_file(self, id_micro):
        """
        Get path to csv file containing epsilon and sigma mean values
        """
        return '{}/{}_epsilon_sigma.csv'.format(self._get_path_csv(id_micro), id_micro)

    def _plot_loading_curve(self, epsilon, sigma, id_micro, path_loading_curve):
        plt.plot(epsilon, sigma, label=(str(id_micro)))
        plt.title('Loading curve')
        plt.xlabel('Epsilon_xx')
        plt.ylabel('Sigma_xx (GPa)')
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                plt.savefig(path_loading_curve)
                done = True

    def _plot_loading(self, epsilon, sigma, id_micro):
        plt.plot(epsilon, sigma, label=(str(id_micro)))
        plt.xlabel('Epsilon_xx', fontsize=14)
        plt.ylabel('Sigma_xx (GPa)', fontsize=14)
        plt.grid()

    def _get_path_loading_curve(self, id_micro):
        return self._get_path_loading(id_micro) + '/loading_curve_{}.png'.format(id_micro)

    def _get_path_loading_curve_elas(self, id_micro, direction_loading, direction_observed):
        """
        Get loading curve path for elasticity results
        """
        return self._get_path_elas(id_micro, direction_loading) + '/loading_curve_{}_{}.png'.format(direction_loading, direction_observed)

    def _fetch_sigma_cmd(self, step, id_micro, micro_shape, N_STEP, direction):
        """
        Command to fetch sigma value
        """
        return "./mstats -s {} {} {} {}{}_Sigma_{} 2>&1|grep -v Re|sed 's/.* //g'".format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_name_file(id_micro), self._convert_to_morphhom_format(step, N_STEP), direction)

    def _fetch_epsilon_cmd(self, step, id_micro, micro_shape, N_STEP, direction):
        """
        Command to fetch epsilon value
        """
        return "./mstats -s {} {} {} {}{}_Epsilon_{} 2>&1|grep -v Re|sed 's/.* //g'".format(micro_shape[0], micro_shape[1], micro_shape[2], self._get_name_file(id_micro), self._convert_to_morphhom_format(step, N_STEP), direction)

    def _get_path_3D_lines(self, id_micro):
        """
        Get path to folder containing 3D numpy array file of localization lines
        """
        return self._get_path_image(id_micro) + '/3D_lines'

    def _get_path_3D_lines_image(self, id_micro):
        """
        Get path to npy file of 3D lines
        """
        return self._get_path_3D_lines(id_micro) + '/loc_lines_{}.npy'.format(id_micro)

    def _convert_2D_into_3D(self, id_micro, axis='z'):
        """
        Convert ppm 2D images of localization lines into 3D numpy array.
        """
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                micro_npy = np.load(self._get_path_npy_image(id_micro))
                Nx, Ny, Nz = micro_npy.shape
                print('micro_npy.shape = ', micro_npy.shape)
                merge_image = np.zeros((Nx, Ny, Nz, 4))
                if axis == 'z':
                    for id_z in range(1, Nz + 1):
                        img_cs = imread(self._get_path_ppm_image(id_micro, id_z, micro_npy.shape))
                        merge_image[:, :, id_z - 1, :3] = np.transpose(img_cs, axes=[1, 0, 2])

                else:
                    if axis == 'x':
                        for id_x in range(1, Nx + 1):
                            img_cs = imread(self._get_path_ppm_image(id_micro, id_x, micro_npy.shape))
                            print('img_cs.shape = ', img_cs.shape)
                            print('np.transpose(img_cs).shape = ', np.transpose(img_cs, axes=[1, 0, 2]).shape)
                            merge_image[id_x - 1, :, :, :3] = img_cs

                merge_image[:, :, :, 3] = merge_image[:, :, :, 2]
                np.save(self._get_path_3D_lines(id_micro) + '/loc_lines_{}.npy'.format(id_micro), merge_image)
                done = True

    def _merge_inclusion_localization(self, id_micro):
        """
        Merge inclusion images with localization lines images.
        Load inclusion localization planes file and then merge it with inclusions
        """
        micro_loc = np.load(self._get_path_3D_lines_image(id_micro))
        micro_inclusions = np.load(self._get_path_npy_image(id_micro))
        Nx, Ny, Nz, Nc = micro_loc.shape
        index_matter1 = np.where(micro_inclusions == 2)
        index_matter2 = np.where(micro_inclusions == 3)
        micro_loc[index_matter1] = [0, 1, 0, 0]
        micro_loc[index_matter2] = [0, 0, 1, 0]
        print('micro_loc = ', micro_loc)
        for id_x in range(1, Nx + 1):
            matplotlib.image.imsave(self._get_path_cs_png(id_micro) + '/cs_{}.png'.format(self._convert_to_morphhom_format(id_x, Nx)), micro_loc[id_x - 1, :, :, :3])

    def _get_path_cs_png(self, id_micro):
        """
        Get path to merged cross section png images
        """
        return self._get_path_image(id_micro) + '/cross_section_png'

    def get_epsilon_sigma_int(self, id_micro):
        """Get epsilon and sigma tab as int numpy arrays"""
        path = self._get_path_csv_file(id_micro)
        with open(path) as (f):
            epsilon = np.asarray(list(map(float, f.readline().split(','))))
            sigma = np.asarray(list(map(float, f.readline().split(','))))
        return (
         epsilon, sigma)

    def plot_from_csv(self, id_micro, positive=True):
        """
        Get epsilon and sigma from csv file and plot them.
        """
        path = self._get_path_csv_file(id_micro)
        with open(path) as (f):
            epsilon = np.asarray(list(map(float, f.readline().split(','))))
            sigma = np.asarray(list(map(float, f.readline().split(','))))
        if positive:
            sigma = abs(sigma)
            epsilon = abs(epsilon)
        self._plot_loading(epsilon, sigma, id_micro)

    def plot_sigma_normalized_from_csv(self, id_micro, positive):
        """
        Plot normalized sigma value from csv file.
        Sigma is divided by its biggest absolute value.
        """
        path = self._get_path_csv_file(id_micro)
        with open(path) as (f):
            epsilon = np.asarray(list(map(float, f.readline().split(','))))
            sigma = np.asarray((list(map(float, f.readline().split(',')))), dtype=(np.float64))
        sigma /= abs(min(sigma))
        if positive:
            sigma = abs(sigma)
            epsilon = abs(epsilon)
        self._plot_loading(abs(epsilon), abs(sigma), id_micro)

    def plot_db_micro(self, tab_names, normalized=True, positive=False):
        """
        Plot results from database of microstructure.
        Ids of microstructures are contained in file_names
        """
        for i in range(len(tab_names)):
            if i % 20 == 0:
                print('plotting {}th curve '.format(i))
            if normalized:
                self.plot_sigma_normalized_from_csv(tab_names[i], positive)
            else:
                self.plot_from_csv(tab_names[i], positive)

    def get_tab_sigma(self, path):
        """
        Get epsilon value of csv file
        path = self._get_path_csv_file(id_micro)
        """
        with open(path, 'r') as (f):
            epsilon = f.readline()
            sigma = f.readline()
        return sigma.split(',')

    def _get_last_sigma(self, path):
        return float(self.get_tab_sigma(path)[(-1)])

    def _get_last_sigma_from_id(self, id_micro):
        """
        Get last sigma value from id micro
        """
        path = self._get_path_csv_file(id_micro)
        with open(path, 'r') as (f):
            epsilon = f.readline()
            sigma = f.readline()
        return sigma.split(',')


    def _write_sigma_to_properties(self, id_micro):
        """
        Write last sigma to properties file
        """
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                last_sigma = self._get_last_sigma(self._get_path_csv_file(id_micro))
                d_properties = pd.read_csv(self._get_path_properties_file(id_micro)).set_index('id_micro')
                d_properties['last_sigma'] = last_sigma
                d_properties.to_csv((self._get_path_properties_file(id_micro)), index=True)
                done = True

    def _write_elasticity_matrix_to_properties(self, id_micro):
        """
        Write 21 unique values of symetric 6x6 elasticity matrix to properties DataFrame.
        """
        with open(self._get_path_elasticity_matrix_csv(id_micro), 'r') as (matrix):
            tab_values = list(map(float, matrix.readlines()))
        d_properties = pd.read_csv(self._get_path_properties_file(id_micro)).set_index('id_micro')
        for i in range(len(tab_values)):
            k, j = self._get_indices_triangular_matrix(i + 1, 6)
            d_properties['M{},{}'.format(k, j)] = tab_values[i]
        else:
            d_properties.to_csv((self._get_path_properties_file(id_micro)), index=True)

    def _add_all_properties(self, id_micro):
        """
        Write last sigma and elasticity matrix to properties file
        """
        self._write_sigma_to_properties(id_micro)
        self._write_elasticity_matrix_to_properties(id_micro)

    def _get_indices_triangular_matrix(self, N, n, i=1):
        """
        Return i,j indices of the n-th element of a triangular nxn matrix
        """
        if N - n <= 0:
            return (i, N + i - 1)
        return self._get_indices_triangular_matrix(N - n, n - 1, i + 1)

    def get_tab_epsilon(self, path):
        """
        Get epsilon tab of csv file
        self._get_path_csv_file(id_micro)
        """
        with open(path, 'r') as (f):
            epsilon = f.readline()
        return epsilon.split(',')

    def _generate_elasticity_matrix(self, id_micro):
        """
        Generate elasticity matrix after average of sigma has been computed.
        The Voigt notation is used.
        """
        done = False
        if self._is_connection_broken():
            time.sleep(5)
        else:
            done = True
            return np.array(list(map(lambda d: self._get_column_elasticity_matrix(id_micro, direction_loading=d), D_VOIGT))).transpose()

    def _get_column_elasticity_matrix(self, id_micro, direction_loading):
        """
        Get column of elasticity matrix with the right direction 
        """
        s_path = list(map(lambda d: self._get_path_elas_file(id_micro, d_loading=direction_loading, d_obs=d), D_VOIGT))
        sigmas = list(map(lambda path: float(self.get_tab_sigma(path)[1]), s_path))
        return sigmas

    def _write_elasticity_matrix(self, id_micro):
        """
        Write elasticity matrix as txt file.
        """
        elas_mat = self._generate_elasticity_matrix(id_micro)
        path_matrix = self._get_path_elasticity_matrix_txt(id_micro)
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                with open(path_matrix, 'w') as (matrix_file):
                    for line in elas_mat:
                        matrix_file.write('   '.join([str(a) for a in line]) + '\n\n')

                done = True

    def _get_path_elasticity_matrix_csv(self, id_micro):
        """
        Get path with the 21 unique values of the elasticity matrix.
        """
        return self._get_path_elasticity_matrix(id_micro) + '/elasticity_matrix.csv'

    def _save_elasticity_matrix(self, id_micro):
        """
        Save elasticity matrix as csv file.
        """
        print('id_micro = ', id_micro)
        elas_matrix = self._generate_elasticity_matrix(id_micro)
        elas_mat_cropped = np.concatenate([elas_matrix[j, j:] for j in range(6)])
        done = False
        while not done:
            if self._is_connection_broken():
                time.sleep(5)
            else:
                np.savetxt((self._get_path_elasticity_matrix_csv(id_micro)), elas_mat_cropped, delimiter=',')
                done = True

    def _del_elasticity_folder(self, id_micro):
        """
        Delete elasticity folder containing intermediary results
        """
        os.system('rm -r {}/{}'.format(PATH_MORPHHOM, id_micro))

    def save_global_properties_file(self, df):
        """
        Save global dataframe df as csv file 
        """
        df.to_csv((self._get_path_global_csv_file()), index=True)

    def plot_linear_curves(self, id_micro):
        """
        Plot linear curves from elasticity and plasticity phases
        """
        epsilon_control, sigma_control = self.get_epsilon_sigma_int(id_micro)
        sigma_control = abs(sigma_control)
        print('sigma_control = ', sigma_control)
        print('epsilon_control = ', epsilon_control)
        epsilon_control = abs(epsilon_control)
        reg0 = linear_model.LinearRegression()
        reg0.fit(sigma_control[:2].reshape(-1, 1), epsilon_control[:2].reshape(-1, 1))
        print('first prediction: ', reg0.predict(epsilon_control.reshape(-1, 1)))
        reg1 = linear_model.LinearRegression()
        reg1.fit(sigma_control[-2:].reshape(-1, 1), epsilon_control[-2:].reshape(-1, 1))
        a0 = reg0.coef_
        b0 = reg0.intercept_
        a1 = reg1.coef_
        b1 = reg1.intercept_
        print('a0 = {}, b0 = {}'.format(a0, b0))
        plt.plot(epsilon_control, reg0.predict(epsilon_control.reshape(-1, 1)), 'x')
        plt.plot(epsilon_control, reg1.predict(epsilon_control.reshape(-1, 1)))

    def _get_precise_elasticity_threshold(self, id_micro):
        """
        Get elasticity threshold Rc by taking the intersection of the linear curves
        corresponding to the linear and plastic phases.
        """
        epsilon_control, sigma_control = self.get_epsilon_sigma_int(id_micro)
        sigma_control = abs(sigma_control)
        reg0 = linear_model.LinearRegression()
        reg0.fit(sigma_control[:2].reshape(-1, 1), epsilon_control[:2].reshape(-1, 1))
        reg1 = linear_model.LinearRegression()
        reg1.fit(sigma_control[-2:].reshape(-1, 1), epsilon_control[-2:].reshape(-1, 1))
        a0 = reg0.coef_
        b0 = reg0.intercept_
        a1 = reg1.coef_
        b1 = reg1.intercept_
        A = np.array([[a0[0][0], -1], [a1[0][0], -1]])
        b = np.array([-b0, -b1])
        x = np.linalg.solve(A, b)
        Rc_sim = -x[0][0]
        return Rc_sim

    def _save_elasticity_threshold(self, id_micro):
        """Compute and save elasticity threshold Rc to properties file"""
        Rc = self._get_precise_elasticity_threshold(id_micro)
        d_properties = pd.read_csv(self._get_path_properties_file(id_micro)).set_index('id_micro')
        d_properties['Rc'] = Rc
        d_properties.to_csv((self._get_path_properties_file(id_micro)), index=True)

    def get_plasticity_threshold(self, id_micro):
        """
        Get plasticity threshold Rc of the microstructure identified by id_micro.
        Rc should have been computed and saved in the properties file of 
        the microstructure before calling this function. 
        """
        d_properties = pd.read_csv(self._get_path_properties_file(id_micro))
        rc_row = d_properties.columns.get_loc('Rc')
        return d_properties.iloc[(0, rc_row)]

    def generate_ids_micro_txt_file(self):
        """
        Generate the txt file containing all of the identifiers of the microstructures of
        the microstructures directory.
        If you need the select only a fraction of the microstructures, you should 
        process the ids variable with some regular expression.
        """
        ids = os.listdir(self._get_path_microstructures_directory())
        print('m = ', m)
        with open('/home/vludvig/nas/stage/python/clustering_v1/micro_fractions.txt', 'w') as (f):
            f.writelines('\n'.join(names))
        print('len(m) = ', len(m))

if __name__ == '__main__':
    fileManager = FileManager()
    id_micro = '1_100_1e-05_50_3_4'

    # fileManager.generate_results(id_micro, N_STEP=100, loading_step=3e-5, void=False)
    #id_micro = '13_406_3e-05_50_357_365'
    #fileManager._generate_images_end(id_micro='1578_100_1e-05_50_3_4', N_STEP=50)
    fileManager.plot_from_csv('1964_100_1e-05_50_3_4')
    fileManager.plot_from_csv('907_100_1e-05_50_3_4')
    plt.show()
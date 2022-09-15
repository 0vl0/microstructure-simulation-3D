import numpy as np
import numpy.random as rdn
from FileManager import FileManager
import abc
from collections import defaultdict

class AbstractMicrostructure(abc.ABC):

    @abc.abstractmethod
    def __init__(self, length, number_forms):
        """
        This class is used to generate a random microstructure containing several inclusions, 
        which is used to make mechanical calculations.
        This version supports only cubic microstructures (only cubic spaces are supported for numerical computations).
        Params:
            length: int, length of the cube, in voxels
            number_forms, int tuple containing number of forms for each phase
        """
        self._length = length
        # matrix material of microstructure = 1
        self._space = np.ones((self._length, self._length, self._length), dtype=np.int32)
        self._inclusion_points = []
        self._number_forms = number_forms
        self._FileManager = FileManager()
        self._number_phase = len(number_forms)
        # tab containing volumic fractions of the different materials
        self._init_fraction_tab()
        self._id_micro = ""

    def _init_fraction_tab(self):
        self._fraction_tab = np.zeros(self._number_phase, dtype=int)

    def _is_form_overlapping(self, points):
        """
        Return True if a point inside of points is already in the microstructure
        """
        return any(self._space[points] > 1)

    def _get_points_insertion(self, starting_point, points_form):
        """
        Get points of the inserted form 
        Inputs:
            starting_point: intxintxint, coordinates of the starting point
            points_form: intxintxint numpy array, array of coordinates
        Output:
            intxintxint numpy array, array of coordinates in the microstructure
        """
        return points_form[0] + starting_point[0], points_form[1] + starting_point[1], points_form[2] + starting_point[2]

    def _modulo_point(self, point):
        """
        Return point modulo dimension of the 3D space. The structure is then symetric.
        Input:
            point: intxintxint, 3D point
        Output:
            intxintxint, point%dimension
        """
        return point[0]%self._length, point[1]%self._length, point[2]%self._length
    
    
    def _find_random_empty_point(self):
        """
        Find a point not yet occupied by an inclusion. The process is random.
        """
        random_point = rdn.randint((self._length, self._length, self._length))
        while self._space[random_point[0], random_point[1], random_point[2]] != 1:
            random_point = rdn.randint((self._length, self._length, self._length))
        return random_point
    
    def _insert_form_random(self, points_form, phase):
        """
        Insert a 3D inclusion inside of the micrsotructure at random position
        """
        inserted = False
        n_try = 0
        while not inserted:
            n_try += 1
            if n_try % 1000 == 0:
                print("n_try = ", n_try)
            if n_try > 30000:
                print("n_try > 30 000, restarting calculation..")
                n_try = 0
                #self.generate_microstructure(self._id_micro)
                # the last form couldn't be easily inserted, so an other form
                # is randomly chosen
                self._insert_form_random(self._get_random_form(), phase)
                inserted=True
            starting_point = self._find_random_empty_point()
            points_insertion = self._get_points_insertion(starting_point, points_form)
            points_insertion = self._modulo_point(points_insertion)
            if not self._is_form_overlapping(points_insertion):
                inserted = True
                n_try = 0
                # update number of points of the corresponding phase
                self._fraction_tab[phase-2] += len(points_insertion[0])
        self._space[points_insertion] = phase

    def _get_random_form(self):
        random_form = self._FileManager.get_random_image()
        return np.where(random_form==1)

    @abc.abstractmethod
    def _insert_inclusion(self):
        """
        Insert inclusion and save the corresponding 3D field as an image identified by i.
        Input:
            i: int, id of the 3D image generated
        """
        """for _ in range(self._number_forms):
            for p in range(2, self._number_phase+2):
                random_form = self._FileManager.get_random_image()
                random_form = np.where(random_form==1)
                self._insert_form_random(random_form, p)"""
    
    def generate_microstructure(self, id_micro):
        """
        Generate and save a random microstructure
        """
        self._id_micro = ""
        self._reset_space()
        self._insert_inclusion()
        self._save_results(id_micro)

    def _reset_space(self):
        self._space = np.ones((self._length, self._length, self._length))
        self._init_fraction_tab()

    def _save_results(self, id_micro):
        """
        Create microstructure directory and save the spaces.
        """
        self._FileManager._create_micro_directory(id_micro)
        self._FileManager.save_npy(self._space, id_micro)
        self._FileManager.save_raw(self._space, id_micro)
        self._save_volumic_fraction(id_micro)

    def _save_volumic_fraction(self, id_micro):
        """
        Save volumic fraction in the properties file
        """
        d = {}
        d["id_micro"] = id_micro
        for i in range(self._number_phase):
            d["f_{}".format(i+2)] = (self._fraction_tab[i]/(self._length)**3)
        self._FileManager.save_properties_file(id_micro, d)  
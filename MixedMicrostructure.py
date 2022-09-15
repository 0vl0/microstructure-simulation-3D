import numpy as np
import numpy.random as rdn
from AbstractMicrostructure import AbstractMicrostructure

class MixedMicrostructure(AbstractMicrostructure):
    def __init__(self, length, number_forms):
        """
        Microstructure which contains different number of each phase
        number_form: tuple with number of forms for each phase
        """
        super().__init__(length, number_forms)

    def _insert_inclusion(self):
        """
        Insert inclusion and save the corresponding 3D field as an image identified by i.
        Input:
            i: int, id of the 3D image generated
        """
        # phase with 0 inclusion are already done
        phase_done_tab = [x == 0 for x in self._number_forms]
        inclusion_inserted = [0] * len(self._number_forms)
        n_inserted = 0
        while not all(phase_done_tab):
            for i in range(len(self._number_forms)):
                print("n_inserted = ", n_inserted)
                if not phase_done_tab[i]:
                    random_form = self._get_random_form()
                    self._insert_form_random(random_form, i+2)
                    inclusion_inserted[i] += 1
                    n_inserted += 1
                    if inclusion_inserted[i] == self._number_forms[i]:
                        phase_done_tab[i] = True

    def set_number_forms(self, number_forms):
        self._number_forms = number_forms

if __name__ == "__main__":
    MixedMicro = MixedMicrostructure(100, (4,6))
    #MixedMicro.generate_microstructure("100_test_3")
    #MixedMicro.generate_microstructure("100_test_4")
    MixedMicro.generate_microstructure("test_4_6")
 

        

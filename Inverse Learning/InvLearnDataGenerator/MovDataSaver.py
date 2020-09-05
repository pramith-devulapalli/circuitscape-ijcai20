import numpy as np
import os
import pandas as pd


class MovDataSaver:
    def __init__(self, filenames=None, file_dir=None, animal_mov_dataframes=None, animal_names=None, max_trap=None):
        self.file_dir = file_dir
        self.filenames = filenames
        self.animal_mov_dataframes = animal_mov_dataframes
        self.animal_names = animal_names
        self.max_trap = max_trap

    def save_movement_data(self, animal_mov_dataframes, animal_names, max_trap, filenames, file_dir=None):
        self.save_auxiliary_data(animal_names, max_trap, filenames[1], file_dir=file_dir)
        self.save_dataframes(animal_mov_dataframes, filenames[0], file_dir=file_dir)

    def save_auxiliary_data(self, animal_names, max_trap, filename, file_dir=None):
        file_path = ''
        if file_dir is None:
            file_path = os.path.normpath(filename)
        else:
            file_path = os.path.normpath(os.path.join(file_dir, filename))
        animal_names.append(max_trap)
        np.save(file_path, np.array(animal_names, dtype=object))

    def save_dataframes(self, animal_mov_dataframes, filename, file_dir=None):
        file_path = ''
        if file_dir is None:
            file_path = os.path.normpath(filename + '.csv')
        else:
            file_path = os.path.normpath(os.path.join(file_dir, filename + '.csv'))
        result = pd.concat(animal_mov_dataframes)
        result.to_csv(file_path, index=False)

    def __call__(self, *args, **kwargs):
        self.save_movement_data(self.animal_mov_dataframes, self.animal_names, self.max_trap, self.filenames,
                                file_dir=self.file_dir)


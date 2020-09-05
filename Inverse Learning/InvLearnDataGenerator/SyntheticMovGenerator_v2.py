import sys
import numpy as np
import argparse
import pandas as pd
from collections import OrderedDict
import datetime
import time
from copy import deepcopy
from InverseLearning.InvLearnDataGenerator.MovDataSaver import MovDataSaver


class SyntheticMovGenerator:
    def __init__(self, num_timesteps=None, num_animals=None, res_matrix=None, matrix_params=None, filenames=None,
                 file_dir=None, seed=42):
        self.num_timesteps = num_timesteps
        self.num_animals = num_animals
        self.res_matrix = res_matrix
        self.matrix_params = matrix_params
        self.filenames = filenames
        self.file_dir = file_dir
        self.seed = seed

    # This creates the full animal movement data
    def generate_movement_from_res_matrix(self, num_timesteps, num_animals, res_matrix=None, seed=42):
        np.random.seed(seed)
        #res_matrix_1 = self.resistance_to_conductance(res_matrix)
        res_matrix_1 = res_matrix
        #print(res_matrix)
        #print('Conductance Matrix')
        #print(res_matrix_1)
        prob_matrix = self.normalized_res_matrix(np.copy(res_matrix_1))
        prob_matrix = self.arr_set_zero(res_matrix_1, prob_matrix)
        #print('Probability Matrix')
        #print(prob_matrix)
        rand_arr = np.random.rand(num_animals, num_timesteps-1)
        animal_names_entries, animal_names = self.generate_animal_names(num_animals, num_timesteps)
        od_list = self.initialize_ordered_dict(animal_names_entries)
        i = 0
        cam_trap = []
        #print('Randomly Generated Array:')
        #print(rand_arr)
        for animal_mov in rand_arr:
            #print('One animal movement')
            #print(animal_mov)
            # Starting point, randomly selected integer
            cam_trap.append(np.random.randint(res_matrix_1.shape[0]))
            #print(cam_trap)
            for mov in animal_mov:
                #print('Movement: ', mov)
                #print('Selected probability matrix: ', prob_matrix[:, cam_trap[-1]])
                # Axis 0
                #cam_trap.append(self.select_next_node(prob_matrix[:, cam_trap[-1]], mov))
                # Axis 1
                cam_trap.append(self.select_next_node(prob_matrix[cam_trap[-1]], mov))
            self.add_time_and_camera(od_list[i], cam_trap, num_timesteps)
            cam_trap.clear()
            i = i + 1
        animal_mov_dataframes = self.dict_to_dataframe(od_list)
        return animal_mov_dataframes, animal_names, res_matrix_1.shape[0]


    def create_partial_trajectories(self, animal_mov_dataframes, num_cam_traps, max_trap):
        #print(max_trap)
        #print(num_cam_traps)
        #print(animal_mov_dataframes)
        sparse_traps = np.random.choice(max_trap, num_cam_traps, replace=False)
        #print(sparse_traps)
        #sparse_mov_dataframes = []
        '''
        for df in animal_mov_dataframes:
            a = df[df['Camera Trap'].isin(sparse_traps)]
            print(a)
            #sparse_mov_dataframes.append(a)
        '''
        sparse_mov_dataframes = [df[df['Camera Trap'].isin(sparse_traps)] for df in animal_mov_dataframes]
        return sparse_mov_dataframes
        #return animal_mov_dataframes

    def normalized_res_matrix(self, res_matrix):
        norm = np.sum(res_matrix, axis=1).reshape(-1, 1)
        # Axis 0
        #norm = np.sum(res_matrix, axis=0)
        return res_matrix/norm

    def arr_set_zero(self, res_matrix, prob_matrix):
        indices = np.nonzero(res_matrix)
        # Axis 0
        #prob_matrix = np.cumsum(prob_matrix, axis=0)
        prob_matrix = np.cumsum(prob_matrix, axis=1)
        mask = np.ones_like(prob_matrix, dtype=bool)
        mask[indices] = False
        prob_matrix[mask] = 0
        return prob_matrix

    def compute_res_matrix(self, weights, res_factors):
        #print(weights)
        #print(res_factors)
        r = sum([x*y for x, y in zip(weights, res_factors)])
        r_symmetric = (r + r.T)/2
        #print('Symmetric matrix:')
        #print(r_symmetric)
        return r_symmetric

    def resistance_to_conductance(self, res_mat):
        res_mat = res_mat.astype('float')
        res_mat[res_mat == 0] = np.inf
        return np.reciprocal(res_mat)

    def select_next_node(self, prob_arr, rand_val):
        a = prob_arr - rand_val
        # Find positive values in the array a
        pos_vals = a[np.where(a >= 0)]
        # Find the minimum of the positive values
        min_val = min(pos_vals)
        # Next camera trap location
        next_cam_trap = np.where(a == min_val)[0][0]
        return next_cam_trap

    def initialize_ordered_dict(self, animal_names_entries):
        od_list = [OrderedDict([('Time', list()), ('Camera Trap', list()), ('Animal Individual', entry)]) for entry in
                   animal_names_entries]
        return od_list

    def dict_to_dataframe(self, od_list):
        return [pd.DataFrame(data=od) for od in od_list]

    def generate_animal_names(self, num_animals, num_timesteps):
        animal_names = ['a' + str(i) for i in range(num_animals)]
        animal_names_entries = [[a]*num_timesteps for a in animal_names]
        return animal_names_entries, animal_names

    def add_time_and_camera(self, od, cam_trap_data, num_timesteps):
        od['Camera Trap'] = deepcopy(cam_trap_data)
        #od['Time'] = self.generate_timestamps(num_timesteps)
        od['Time'] = list(np.arange(start=1, stop=num_timesteps+1))

    def generate_timestamps(self, num_timesteps):
        timestamps = [self.generate_random_timestamp() for _ in range(num_timesteps)]
        timestamps.sort(key=lambda date: datetime.datetime.strptime(date, '%m-%d-%y %H:%M:%S'))
        return timestamps

    def generate_random_timestamp(self):
        val = np.random.random()
        random_int = val*(int(time.time()))
        random_time = datetime.datetime.strftime(datetime.datetime.fromtimestamp(random_int), '%m-%d-%y %H:%M:%S')
        return random_time

    def __call__(self, *args, **kwargs):
        if self.res_matrix is not None and self.matrix_params is None:
            animal_movement_data, animal_names, max_trap = self.generate_movement_from_res_matrix(self.num_timesteps,
                                                                                                  self.num_animals,
                                                                                                  res_matrix=
                                                                                                  self.res_matrix,
                                                                                                  seed=self.seed)
            #sparse_movement_data = self.create_partial_trajectories(animal_movement_data, self.num_cam_traps, max_trap)
            if self.filenames is not None:
                mds = MovDataSaver(filenames=self.filenames, file_dir=self.file_dir,
                                   animal_mov_dataframes=animal_movement_data, animal_names=animal_names,
                                   max_trap=max_trap)
                mds()
            return animal_movement_data, animal_names, max_trap

        elif self.res_matrix is not None and self.matrix_params is not None:
            weights = [np.squeeze(weight) for weight in self.res_matrix]
            #print(weights)
            #self.matrix_params = np.load(self.matrix_params)
            factors = [np.squeeze(factor) for factor in self.matrix_params]
            #print(factors)
            #print(self.compute_res_matrix(weights, factors))
            animal_movement_data, animal_names, max_trap = self.generate_movement_from_res_matrix(self.num_timesteps,
                                                                                                   self.num_animals,
                                                                     res_matrix=self.compute_res_matrix(weights,
                                                                                                        factors),
                                                                                                   seed=self.seed)
            #sparse_movement_data = self.create_partial_trajectories(animal_movement_data, self.num_cam_traps, max_trap)
            if self.filenames is not None:
                mds = MovDataSaver(filenames=self.filenames, file_dir=self.file_dir,
                                   animal_mov_dataframes=animal_movement_data, animal_names=animal_names,
                                   max_trap=max_trap)
                mds()
            return animal_movement_data, animal_names, max_trap

'''
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Generate animal movement data based on a resistance matrix and other'
                                                 ' specifications')
    parser.add_argument("num_timesteps", help="Number of timesteps to synthetically generate animal movement data for "
                                              "each animal.", type=int)
    parser.add_argument("num_animals", help="Number of animals for which data should be generated for.", type=int)
    parser.add_argument("res_matrix", help="File location of either resistance matrix or weight vector. If weight "
                                           "vector is given, then add the optional argument --factor and specify file "
                                           "location of factor parameters.")
    parser.add_argument('sparse_traps', help='Number of camera traps from which animal individual data can be collected'
                                             ' from. This must be less than the dimensions of the resistance matrix. '
                                             'An error will be thrown if this property is violated.', type=int)
    parser.add_argument("--factor", dest="factor", help="File location of resistance factor parameters.",
                        action='store', nargs='?')
    parser.add_argument('--filenames', help="List of filenames to store animal movement data. List one filename for "
                                            "storing animal movement data in CSV format and another filenames for "
                                            "storing auxiliary data.", dest="file", default=None, const=None, nargs='*')
    parser.add_argument('--file_dir', help="File directory to store animal movement data. If not specified, "
                                           "--filenames argument must contain relative or absolute file paths.",
                        dest="dir", default=None, const=None, nargs='?')
    parser.add_argument("--seed", help="Seed of random number generator, automatically defaults to 42.", action='store',
                        default=42, dest='seed', nargs='?', const=42)
    args = parser.parse_args()
    res_matrix_1 = np.load(args.res_matrix)
    smg = SyntheticMovGenerator(num_timesteps=args.num_timesteps, num_animals=args.num_animals,
                                res_matrix=res_matrix_1, matrix_params=args.factor, filenames=args.file,
                                file_dir=args.dir, num_cam_traps=args.sparse_traps, seed=args.seed)
    a_m_d, a_n, max_cam = smg()
    print(a_m_d)
    print(a_n)
    print(max_cam)
'''

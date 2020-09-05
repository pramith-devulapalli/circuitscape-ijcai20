import numpy as np
import torch
import pandas as pd
from MantelCircuitscape.MantelOptCG import MantelOptConjugateGradient
import argparse
import sys
import os
import pickle


class MantelData:
    def __init__(self, res_filenames, genetic_df_name, res_directory, num_animals, BC_distances, num_iterations_cg,
                 dictionary_name, mean=0.05, std=0.015):
        self.res_filenames = res_filenames
        self.genetic_df_name = genetic_df_name
        self.res_directory = res_directory
        self.num_animals = num_animals
        self.BC_distances = BC_distances
        self.num_iterations_cg = num_iterations_cg
        self.dictionary_name = dictionary_name
        self.mean = mean
        self.std = std

    def generate_individuals(self, cond_matrix, num_animals):
        np.random.seed(42)
        locations = cond_matrix.shape[0]
        animal_locations = dict()
        i = 0
        for animal in range(num_animals):
            animal_locations['individual' + str(i)] = np.random.randint(0, high=locations)
            i = i + 1
        return animal_locations

    def read_res_file_paths(self, res_filenames, res_dir=None):
        if res_dir is not None:
            factor_file_path = os.path.normpath(os.path.join(res_dir, res_filenames[0]))
            weights_file_path = os.path.normpath(os.path.join(res_dir, res_filenames[1]))
        else:
            factor_file_path = os.path.normpath(res_filenames[0])
            weights_file_path = os.path.normpath(res_filenames[1])

        factor_params = np.load(factor_file_path)
        weight_vec = np.load(weights_file_path)
        #print(weights_file_path)

        return factor_params, weight_vec

    def compute_res_matrix(self, weights, res_factors):
        #print(weights)
        #print(res_factors)
        r = sum([x*y for x, y in zip(weights, res_factors)])
        return r

    # I think this function works properly
    def compute_commuting_times(self, model, res_matrix, cg_iterations):
        commuting_time_matrix = torch.zeros(res_matrix.shape)
        with torch.no_grad():
            for i in range(res_matrix.shape[0]):
                for j in range(i+1, res_matrix.shape[1]):
                    commute_time = model(i, j, full_iterations=cg_iterations)
                    commuting_time_matrix[i, j] = commute_time
                    commuting_time_matrix[j, i] = commute_time

        return commuting_time_matrix

    def generate_dataframe_random(self, animal_locations):
        np.random.seed(42)
        key_set = animal_locations.keys()
        key_set_2 = set(animal_locations.keys())
        genetic_df = pd.DataFrame(0.0, columns=list(animal_locations.keys()), index=list(animal_locations.keys()))

        for animal in key_set:
            key_set_2.remove(animal)
            for individual in key_set_2:
                value = np.random.rand()
                genetic_df[individual][animal] = value
                genetic_df[animal][individual] = value

        return genetic_df

    def generate_dataframe_structured(self, animal_locations, commuting_time_matrix, mean, std):
        np.random.seed(42)
        key_set = animal_locations.keys()
        key_set_2 = set(animal_locations.keys())
        genetic_df = pd.DataFrame(0.0, columns=list(animal_locations.keys()), index=list(animal_locations.keys()))
        #print('Commuting Time Matrix')
        #print(commuting_time_matrix)
        #print('')
        max_commuting_time = torch.max(commuting_time_matrix)
        print('Max Commuting Time:')
        print(max_commuting_time)
        print('')
        for animal in key_set:
            key_set_2.remove(animal)
            for individual in key_set_2:
                # Access first camera trap
                first_camera_trap = animal_locations[animal]

                # Access second camera trap
                last_camera_trap = animal_locations[individual]

                # Extract commuting time
                commuting_time = commuting_time_matrix[first_camera_trap, last_camera_trap]

                #print('')
                #print('First individual: ', animal, ' Location: ', first_camera_trap)
                #print('Second individual: ', individual, 'Location: ', last_camera_trap)
                #print('Commuting Time: ', commuting_time)
                #print('')

                # Generate the Bray-Curtis distance
                bc_dissimilarity = self.generate_BC_index(commuting_time.item(), max_commuting_time.item(), mean, std)

                genetic_df[individual][animal] = bc_dissimilarity
                genetic_df[animal][individual] = bc_dissimilarity
                #print('')

        return genetic_df

    def generate_BC_index(self, current_ct, max_ct, mean, std):
        BC = current_ct/max_ct + np.random.normal(mean, std)
        #print('BC Index: ', BC)

        if BC > 1:
            return 1
        elif BC < 0:
            return 0
        else:
            return BC

    def generate_filename(self, filename, file_dir=None):
        save_file_path = ''
        if file_dir is not None:
            save_file_path = os.path.normpath(os.path.join(file_dir, filename))
        else:
            save_file_path = os.path.normpath(filename)

        return save_file_path

    def __call__(self, *args, **kwargs):
        f_params, w_vec = self.read_res_file_paths(self.res_filenames, res_dir=self.res_directory)
        res_matrix = self.compute_res_matrix(w_vec, f_params)
        n = res_matrix.shape[0]

        model = MantelOptConjugateGradient(n, w_vec, f_params)

        animal_locations = self.generate_individuals(res_matrix, self.num_animals)
        #print(animal_locations)
        with torch.no_grad():
            commuting_matrix = self.compute_commuting_times(model, res_matrix, self.num_iterations_cg)

        if self.BC_distances == 'structured':
            genetic_df = self.generate_dataframe_structured(animal_locations, commuting_matrix, self.mean, self.std)
        else:
            genetic_df = self.generate_dataframe_random(animal_locations)

        # Save the dataframe
        genetic_filename = self.generate_filename(self.genetic_df_name, file_dir=self.res_directory)
        genetic_df.to_csv(genetic_filename + '.csv')

        # Save the animal_locations as a hashmap/dict
        animal_locations_filename = self.generate_filename(self.dictionary_name, file_dir=self.res_directory)
        f = open(animal_locations_filename + '.pkl', 'wb')
        pickle.dump(animal_locations, f)
        f.close()


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Generates a genetic distance matrix using the Bray-Curtis method.")
    parser.add_argument('res_filenames', help="List of file locations for resistance factor parameters and associated "
                                              "weight vector. First filename indicates factor parameters. Second "
                                              "filename indicates weight vector.", default=None, const=None, nargs=2)
    parser.add_argument('genetic_df_name', help="Name of the genetic distance dataframe to save. Will be saved in the "
                                                "res_directory as .csv file.")
    parser.add_argument('res_directory', help='Indicate file directory of resistance factor parameters and weight '
                                              'vector. If not needed, specify N/A.', default='N/A')
    parser.add_argument("num_animals", help="Number of animals for which data should be generated for.", type=int)
    parser.add_argument('BC_distances', help='Indicates how Bray-Curtis genetic distances are created. Type in "random"'
                                             ' for truly random generation or "structured" to pre-correlate resistance '
                                             'surface with genetic distances.', type=str, default=None)
    parser.add_argument('num_iterations_cg', help='The maximum number of conjugate gradient iterations allowed.',
                        type=int)
    parser.add_argument('dictionary_name', help="Name of the animal_locations dictionary which stores the name of the "
                                                "animal individual and the location of that individual.", type=str)
    parser.add_argument('--mean',
                        help='The mean of the Gaussian noise added to Bray-Curtis distances for the structured'
                             ' option.', dest='mean', nargs='?', type=float, default=0.05)
    parser.add_argument('--std', help='The standard deviation of the Gaussian noise added to Bray-Curtis distances for '
                                      'the structured option.', dest='std', nargs='?', type=float, default=0.015)
    args = parser.parse_args()
    m_e = MantelData(args.res_filenames, args.genetic_df_name, args.res_directory, args.num_animals,
                           args.BC_distances, args.num_iterations_cg, args.dictionary_name, mean=args.mean,
                     std=args.std)
    m_e()

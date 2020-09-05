import numpy as np
import torch
import pandas as pd
import argparse
import sys
import os
from MantelCircuitscape.MantelOptCG import MantelOptConjugateGradient
import pickle
import time


class MantelExperiment:
    def __init__(self, res_filenames, starting_weights, res_directory, genetic_csv_name, dictionary_name,
                 num_iterations_cg, epochs, csv_file=None):
        self.res_filenames = res_filenames
        self.starting_weights = starting_weights
        self.res_directory = res_directory
        self.genetic_csv_name = genetic_csv_name
        self.dictionary_name = dictionary_name
        self.num_iterations_cg = num_iterations_cg
        self.epochs = epochs
        self.csv_file = csv_file

    def read_res_file_paths(self, res_filenames, res_dir=None):
        if res_dir is not None:
            factor_file_path = os.path.normpath(os.path.join(res_dir, res_filenames[0]))
            weights_file_path = os.path.normpath(os.path.join(res_dir, res_filenames[1]))
        else:
            factor_file_path = os.path.normpath(res_filenames[0])
            weights_file_path = os.path.normpath(res_filenames[1])

        factor_params = np.load(factor_file_path)
        weight_vec = np.load(weights_file_path)

        return factor_params, weight_vec

    def compute_res_matrix(self, weights, res_factors):
        # print(weights)
        # print(res_factors)
        r = sum([x * y for x, y in zip(weights, res_factors)])
        return r

    def compute_mean_triu(self, array):
        mean = 0
        n = array.shape[0]
        d = n * (n - 1) / 2.0
        for i in range(array.shape[0]):
            for j in range(i + 1, array.shape[1]):
                mean = mean + array[i, j]
        return mean / d

    def compute_std_triu(self, array, mean):
        variance = 0
        for i in range(array.shape[0]):
            for j in range(i + 1, array.shape[0]):
                variance = variance + (array[i, j] - mean) ** 2

        n = array.shape[0]
        d = n * (n - 1) / 2.0

        return np.sqrt(variance / d)

    def compute_mean_triu_torch(self, array):
        mean = torch.DoubleTensor(1, 1)
        mean[0] = 0
        n = array.shape[0]
        d = n * (n - 1) / 2.0
        for i in range(array.shape[0]):
            for j in range(i + 1, array.shape[1]):
                mean = mean + array[i, j]

        return mean / d

    def compute_std_triu_torch(self, array, mean):
        variance = torch.DoubleTensor(1, 1)
        variance[0] = 0
        for i in range(array.shape[0]):
            for j in range(i + 1, array.shape[1]):
                variance = variance + (array[i, j] - mean) ** 2

        n = array.shape[0]
        d = n * (n - 1) / 2.0
        print('d: ', d)

        return torch.sqrt(variance / d)

    def compute_commuting_matrix(self, model, genetic_df, animal_locations, cg_iterations):
        commuting_time_matrix = torch.zeros(genetic_df.values.shape, dtype=torch.float64)
        df_cols = list(genetic_df.columns)
        k = 0
        l = 1

        for i in genetic_df.index:
            df_cols.remove(i)
            for j in df_cols:
                first_cam_trap = animal_locations[i]
                last_cam_trap = animal_locations[j]
                if first_cam_trap != last_cam_trap:
                    commute_time = model(first_cam_trap, last_cam_trap, full_iterations=cg_iterations)
                else:
                    commute_time = 0.0

                commuting_time_matrix[k, l] = commute_time
                commuting_time_matrix[l, k] = commute_time

                l = l + 1

            k = k + 1
            l = k + 1
        return commuting_time_matrix

    def one_epoch_gradient_ascent(self, model, genetic_df, animal_locations, cg_iterations, best_matrix, loss_dict, epoch,
                                  save_file=None, best_matrix_file=None):
        '''
                Before hand, precompute the mean and standard deviation of the genetic distances.
                1. Compute all the commuting times and store them in a 2-d tensor.
                2. Compute the mean and standard deviation of the commuting times and genetic distances.
                3. Compute the Mantel coefficient
                4. Backpropagate through the Mantel coefficient using Adam Optimizer
                '''

        # Have to extract the upper triangular part of the array

        # Precompute the standard deviation and mean of the genetic distance data
        genetic_array = genetic_df.values
        # print(genetic_array)
        genetic_mean = self.compute_mean_triu(genetic_array)
        print('Genetic Mean: ', genetic_mean)
        genetic_std = self.compute_std_triu(genetic_array, genetic_mean)
        print('Genetic Standard Deviation: ', genetic_std)

        start_time = time.perf_counter()

        adam_optimizer = torch.optim.Adam(model.parameters(), lr=2.0)
        adam_optimizer.zero_grad()

        commuting_matrix = self.compute_commuting_matrix(model, genetic_df, animal_locations, cg_iterations)
        # print('Commuting Matrix: ', commuting_matrix)

        mean_commuting_time = self.compute_mean_triu_torch(commuting_matrix)
        print('Mean Commuting Time: ', mean_commuting_time)
        std_commuting_time = self.compute_std_triu_torch(commuting_matrix, mean_commuting_time)
        print('Std Commuting Time: ', std_commuting_time)

        n = commuting_matrix.shape[0]
        d = n * (n - 1) / 2.0
        #mantel_coefficient = torch.tensor(0.0, dtype=torch.float64)
        mantel_coefficient = torch.DoubleTensor(1, 1)
        mantel_coefficient[0] = 0

        #mantel_coefficient = mantel_coefficient.detach()

        df_cols = list(genetic_df.columns)
        indices = {}
        for i, j in enumerate(df_cols):
            indices[j] = i

        for i in genetic_df.index:
            df_cols.remove(i)
            for j in df_cols:
                # Assignment, +=, does something in PyTorch, read more about it

                mantel_coefficient = mantel_coefficient + (1.0 / d) * (
                            (genetic_df[i][j] - genetic_mean) / genetic_std) * ((commuting_matrix[
                                                                                     indices[i], indices[
                                                                                         j]] - mean_commuting_time) / std_commuting_time)

                #mantel_coefficient += model(animal_locations[i], animal_locations[j], full_iterations=cg_iterations)

        print('Mantel Coefficient: ', mantel_coefficient.item())
        print('')
        print('')

        unnegated_coefficient = mantel_coefficient.clone().detach()

        # Back-propagate for gradient ascent
        mantel_coefficient = mantel_coefficient * -1.0
        # mantel_coefficient.backward(retain_graph=True)

        mantel_coefficient.backward()
        adam_optimizer.step()
        # adam_optimizer.step()
        # mantel_coefficient.backward()

        with torch.no_grad():
            for param in model.parameters():
                param[param < 0.0] = 0.0

        end_time = time.perf_counter()
        print('Train time: ', end_time - start_time)
        print('')

        if best_matrix is not None:
            if len(best_matrix) == 0:
                with torch.no_grad():
                    best_matrix.append(unnegated_coefficient.item())
                    best_matrix.append(model.print_weights_vector())
            else:
                if best_matrix[0] < unnegated_coefficient.item():
                    with torch.no_grad():
                        best_matrix[0] = unnegated_coefficient.item()
                        best_matrix[1] = model.print_weights_vector()

        if loss_dict is not None:
            ep = loss_dict['Epochs']
            ep.append(epoch)
            loss_dict['Epochs'] = ep
            ti = loss_dict['Time']
            ti.append(end_time - start_time)
            me = loss_dict['Mantel Coefficient']
            me.append(unnegated_coefficient.item())

        if save_file is not None:
            df = pd.DataFrame(data=loss_dict)
            f = open(save_file, 'w')
            cmd_args = str(self.res_filenames) + ' ' + str(self.starting_weights) + ' ' + str(self.res_directory) + \
                        ' ' + str(self.dictionary_name) + ' ' + str(self.num_iterations_cg) + ' ' + str(self.epochs)

            f.write(cmd_args)
            df.to_csv(f)
            np.save(best_matrix_file, best_matrix)

    def generate_filename(self, filename, file_dir=None):
        save_file_path = ''
        if file_dir is not None:
            save_file_path = os.path.normpath(os.path.join(file_dir, filename))
        else:
            save_file_path = os.path.normpath(filename)

        return save_file_path

    def __call__(self, *args, **kwargs):
        # Read in the base matrices and weight vector
        f_params, w_vec = self.read_res_file_paths(self.res_filenames, res_dir=self.res_directory)
        print(w_vec)
        # Read in the genetic dataframe
        genetic_df = pd.read_csv(os.path.normpath(os.path.join(self.res_directory, self.genetic_csv_name + '.csv')),
                                 index_col=0)

        # Read in the animal locations
        animal_locations_filename = self.generate_filename(self.dictionary_name, file_dir=self.res_directory)
        read = open(animal_locations_filename + '.pkl', "rb")
        animal_locations = pickle.load(read)

        # Read in the starting weight vector
        w_vec_starting = np.load(os.path.normpath(os.path.join(self.res_directory, self.starting_weights[0])))

        print(w_vec_starting)
        # Compute res_matrix
        res_matrix = self.compute_res_matrix(w_vec, f_params)
        print(res_matrix.shape)

        # Extract dimension
        n = res_matrix.shape[0]

        save_file = None
        loss_dict = None
        best_matrix_file = None
        best_matrix = []
        if self.csv_file is not None:
            save_file = os.path.normpath(os.path.join(self.res_directory, self.csv_file + '.csv'))
            best_matrix_file = os.path.normpath(os.path.join(self.res_directory, self.csv_file + '_best_matrix'))
            loss_dict = {'Epochs': [], 'Time': [], 'Mantel Coefficient': []}

        # Initialize a model with the base matrices and starting weight vector
        model = MantelOptConjugateGradient(n, w_vec_starting, f_params)

        for epoch in range(self.epochs):
            print('Epoch: ', epoch)
            self.one_epoch_gradient_ascent(model, genetic_df, animal_locations, self.num_iterations_cg, best_matrix,
                                           loss_dict, epoch, save_file=save_file, best_matrix_file=best_matrix_file)

        # self.gradient_ascent(self.epochs, model, genetic_df, animal_locations, self.num_iterations_cg, best_matrix,
        # loss_dict, save_file=save_file, best_matrix_file=best_matrix_file)

        print('Final Weights: ')
        print(model.print_weights_vector())


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Executes the Mantel Coefficient Optimization experiment given a '
                                                 'genetic distance matrix and a resistance surface.')
    parser.add_argument('res_filenames', help="List of file locations for resistance factor parameters and associated "
                                              "weight vector. First filename indicates factor parameters. Second "
                                              "filename indicates weight vector.", default=None, const=None, nargs=2)
    parser.add_argument('starting_weights', help="File name for weight vector to initialize the machine learning "
                                                 "training process.", nargs=1)
    parser.add_argument('res_directory', help='Indicate file directory of resistance factor parameters and weight '
                                              'vector and genetic distance matrix. If not needed, specify N/A.',
                        default='N/A')
    parser.add_argument('genetic_csv_name', help='Name of the genetic distance matrix file saved as a csv. No need to '
                                                 'add .csv to filename.')
    parser.add_argument('dictionary_name', help="Name of the animal_locations dictionary which stores the name of the "
                                                "animal individual and the location of that individual.", type=str)
    parser.add_argument('num_iterations_cg', help='The maximum number of conjugate gradient iterations allowed.',
                        type=int)
    parser.add_argument('epochs', help='Number of epochs to run training for optimization.', type=int)
    parser.add_argument('--csv_file', help="Name of output file of mantel coefficient and timing performance for every "
                                           "epoch. Header contains command line argument to generate experiment. Will be"
                                           " saved in res_directory.", dest='cf')
    args = parser.parse_args()
    m_e = MantelExperiment(args.res_filenames, args.starting_weights, args.res_directory, args.genetic_csv_name,
                           args.dictionary_name, args.num_iterations_cg, args.epochs, csv_file=args.cf)
    m_e()

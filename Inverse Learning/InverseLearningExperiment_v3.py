import argparse
import sys
import os
import numpy as np
import torch
import pandas as pd
import time
from InvLearnDataGenerator.SyntheticMovGenerator_v2 import SyntheticMovGenerator
from InvLearnDataset.DatasetTransformer_v2 import DatasetTransformer
from InvLearnDataset.Dataset import Dataset
from InvLearnCircuitscape.InvLearnLaplacian import InvLearnDirectLaplacian
from InvLearnCircuitscape.InvLearnCG import InvLearnConjugateGradient


'''
This version 3 includes KL-divergence calculations after every epoch and stores it into the csv file. Additionally, it
prevents any of the values in the weight vector going below 0. 
'''
class InverseLearning:
    def __init__(self, res_filenames, res_directory, num_timesteps, num_animals, cam_traps, grd, epochs, batch_size, mov_filenames=None, file_dir=None,
                 starting_vec=None, ratio=None, learning_rate=None, csv_file=None, cg_iterations=None, seed=42):
        self.res_filenames = res_filenames
        self.res_directory = res_directory
        self.num_timesteps = num_timesteps
        self.num_animals = num_animals
        self.cam_traps = cam_traps
        self.grad_type = grd
        self.epochs = epochs
        self.batch_size = batch_size
        self.mov_filenames = mov_filenames
        self.file_dir = file_dir
        self.starting_vec = starting_vec
        self.ratio = ratio
        self.lr = learning_rate
        self.csv_file = csv_file
        self.cg_iterations = cg_iterations
        print('CG Iterations: ')
        print(self.cg_iterations)
        self.seed = seed

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
        r_symmetric = (r + r.T)/2
        return r_symmetric

    def transform_dataset(self, animal_movement_data, max_trap, num_traps, batch_size, ratio):
        dt = DatasetTransformer(animal_mov_dataframes=animal_movement_data, max_cam_trap=max_trap,
                                num_cam_traps=num_traps, ratio=ratio)
        df_results = dt()
        train_dataset_validation = Dataset(hitting_time_data=df_results[0], batch_size=1)
        train_dataset = Dataset(hitting_time_data=df_results[0], batch_size=batch_size)
        #test_dataset = Dataset(hitting_time_data=df_results[0], batch_size=1)
        test_dataset = Dataset(hitting_time_data=df_results[1], batch_size=1)
        '''
        print('Train')
        for train in train_dataset:
            print(train)

        print('Test')
        for test in test_dataset:
            print(test)
        '''
        return train_dataset.return_iterator(), test_dataset.return_iterator(), train_dataset_validation.return_iterator()

    '''
    def weights_on_dataset(self, train_dataset, res_matrix_like):
        # data in train_data
        # data[2] = number of steps
        # data[1] = last camera trap
        # data[0] = first camera trap
        # The individual indices of data are tensors
        freq_matrix = torch.zeros(res_matrix_like.shape, dtype=torch.float64)
        weights_matrix = torch.zeros(res_matrix_like.shape, dtype=torch.float64)
        for data in train_dataset:
            #print(data)
            freq_matrix[data[0].type(torch.int64), data[1].type(torch.int64)] += 1

        #print('Frequency Matrix: ')
        #print(freq_matrix)
        F = torch.sum(freq_matrix)
        #print('F: ', F)
        for i in range(freq_matrix.shape[0]):
            for j in range(freq_matrix.shape[1]):
                weights_matrix[i, j] = torch.sqrt((freq_matrix[i, j]**2)/(torch.sum(freq_matrix[i])*F))

        #print('Weights Matrix: ')
        #print(weights_matrix)

        return weights_matrix
        '''


    def compute_test_train_loss(self, train_data, test_data, inv_model, epoch_num, loss_dict, time, best_matrix,
                                cg_iterations=None, save_file=None, best_matrix_file=None):
        #print('Start Computing Test Train Loss')
        #print(inv_model)
        print('Length of Test Data: ', len(test_data))
        print('Length of Train Data: ', len(train_data))
        with torch.no_grad():
            #for param in inv_model.parameters():
                #print(param)
            # Train Loss

            tr_loss = 0.0
            #tr_loss_weighted = 0.0
            count = 0
            for data in train_data:
                #print('Data point: ', data)
                #print('Weight matrix value: ', weights_matrix[data[0], data[1]])
                #if count % 10 == 0:
                    #print('Sample data: ', data)
                    #print('First: ', data[0], ' Last: ', data[1], ' Steps: ', data[2])
                    #print('Count: ', count)
                count = count + 1
                a = torch.DoubleTensor(1)
                a[0] = data[2]
                if cg_iterations is None:
                    mse = (a[0] - inv_model(data[0], data[1])).pow(2)
                    #tr_loss += (a[0] - inv_model(data[0], data[1])).pow(2)
                    tr_loss += mse
                    #tr_loss_weighted += weights_matrix[data[0], data[1]]*(a[0] - inv_model(data[0], data[1])).pow(2)
                    #tr_loss_weighted += weights_matrix[data[0], data[1]]*mse
                else:
                    mse = (a[0] - inv_model(data[0], data[1], full=inv_model.return_dimension())).pow(2)
                    #tr_loss += (a[0] - inv_model(data[0], data[1], full=inv_model.return_dimension())).pow(2)
                    tr_loss += mse
                    #tr_loss_weighted += weights_matrix[data[0], data[1]] * (a[0] - inv_model(data[0], data[1])).pow(2)
                    #tr_loss_weighted += weights_matrix[data[0], data[1]]*mse
            print('Training Loss: ', tr_loss.item()/len(train_data))
            #print('Weighted Training Loss: ', tr_loss_weighted.item()/len(train_data))

            # Test Loss
            te_loss = 0.0
            #te_loss_weighted = 0.0
            for data in test_data:
                a = torch.DoubleTensor(1)
                a[0] = data[2]
                if cg_iterations is None:
                    mse = (a[0] - inv_model(data[0], data[1])).pow(2)
                    #te_loss += (a[0] - inv_model(data[0], data[1])).pow(2)
                    te_loss += mse
                    #te_loss_weighted += weights_matrix[data[0], data[1]]*mse
                else:
                    mse = (a[0] - inv_model(data[0], data[1], full=inv_model.return_dimension())).pow(2)
                    #te_loss += (a[0] - inv_model(data[0], data[1], full=inv_model.return_dimension())).pow(2)
                    te_loss += mse
                    #te_loss_weighted += weights_matrix[data[0], data[1]]*mse
            print('Testing Loss: ', te_loss.item()/len(test_data))
            #print('Weighted Testing Loss: ', te_loss_weighted.item()/len(test_data))

            if best_matrix is not None:
                if len(best_matrix) == 0:
                    best_matrix.append(te_loss.item()/len(test_data))
                    best_matrix.append(inv_model.print_weights_vector())
                else:
                    if best_matrix[0] > te_loss.item()/len(test_data):
                        best_matrix[0] = te_loss.item()/len(test_data)
                        best_matrix[1] = inv_model.print_weights_vector()

            if loss_dict is not None:
                ep = loss_dict['Epochs']
                ep.append(epoch_num)
                loss_dict['Epochs'] = ep
                tl = loss_dict['Training Loss']
                tl.append(tr_loss.item()/len(train_data))
                loss_dict['Training Loss'] = tl
                te = loss_dict['Testing Loss']
                te.append(te_loss.item()/len(test_data))
                loss_dict['Testing Loss'] = te
                ti = loss_dict['Time']
                ti.append(time)
                loss_dict['Time'] = ti
                kl = loss_dict['KL']
                kl.append(0.0)
                loss_dict['KL'] = kl

            if save_file is not None:
                df = pd.DataFrame(data=loss_dict)
                f = open(save_file, 'w')
                cmd_args = str(self.res_filenames) + ' ' + str(self.starting_vec) + ' ' + str(self.res_directory) + \
                           ' ' + str(self.num_timesteps) + ' ' + str(self.num_animals) + ' ' + str(
                    self.cam_traps) + ' ' + \
                           str(self.grad_type) + ' ' + str(self.epochs) + ' ' + str(self.batch_size) + ' ' + str(
                    self.ratio) \
                           + ' ' + str(self.lr) + ' ' + str(self.csv_file) + ' ' + str(self.cg_iterations) + '\n'
                f.write(cmd_args)
                df.to_csv(f)
                np.save(best_matrix_file, best_matrix)

    def compute_KL_divergence(self, loss_dict, best_matrix, true_w_vec, f_params, save_file=None):
        # Compute the true conductance matrix
        true_cond_matrix = self.add_epsilon(self.compute_res_matrix(true_w_vec, f_params))
        true_prob_matrix = self.normalized_res_matrix(np.copy(true_cond_matrix))
        #true_prob_matrix = self.arr_set_zero(true_cond_matrix, true_prob_matrix)

        # Compute the predicted conductance matrix
        learnt_w_vector = list(best_matrix[1].detach().numpy())
        predicted_cond_matrix = self.add_epsilon(self.compute_res_matrix(learnt_w_vector, f_params))
        learnt_prob_matrix = self.normalized_res_matrix(predicted_cond_matrix)
        #learnt_prob_matrix = self.arr_set_zero(predicted_cond_matrix, learnt_prob_matrix)


        # Calculate KL-Divergence
        KL_divergence = self.calculate_KL(true_prob_matrix, learnt_prob_matrix)
        print('KL-Divergence:')
        print(KL_divergence)

        # Save the value
        if loss_dict is not None:
            kl = loss_dict['KL']
            kl[-1] = KL_divergence
            loss_dict['KL'] = kl

        # Update the csv file
        if save_file is not None:
            df = pd.DataFrame(data=loss_dict)
            f = open(save_file, 'w')
            cmd_args = str(self.res_filenames) + ' ' + str(self.starting_vec) + ' ' + str(self.res_directory) + \
                       ' ' + str(self.num_timesteps) + ' ' + str(self.num_animals) + ' ' + str(
                self.cam_traps) + ' ' + \
                       str(self.grad_type) + ' ' + str(self.epochs) + ' ' + str(self.batch_size) + ' ' + str(
                self.ratio) \
                       + ' ' + str(self.lr) + ' ' + str(self.csv_file) + ' ' + str(self.cg_iterations) + '\n'
            f.write(cmd_args)
            df.to_csv(f)

    def calculate_KL(self, true_matrix, learnt_matrix):
        KL_matrix = true_matrix*np.log(true_matrix/learnt_matrix)
        return np.sum(KL_matrix)

    def add_epsilon(self, cond_matrix):
        cond_matrix = cond_matrix + 0.00000001
        return cond_matrix

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

    def one_epoch_sgd(self, train_data, i_model, cg_iterations=None, learning_rate=0.005):
        #print('')
        #print('Length of training data: ', len(train_data))
        #print('')
        i = 0
        for data in train_data:
            #print('i: ', i)
            print('Data point: ', data)
            if i == 0:
                full_1 = True
            else:
                full_1 = False
            i = i + 1
            i_model.zero_grad()
            #print(i_model)
            if cg_iterations is None:
                h_pred = i_model(data[0], data[1])
            else:
                h_pred = i_model(data[0], data[1], full=cg_iterations)
            #print('Prediction: ', h_pred)
            a = torch.DoubleTensor(1)
            a[0] = data[2]
            loss = (a[0] - h_pred).pow(2)
            #print(loss)
            loss.backward()

            with torch.no_grad():
                for param in i_model.parameters():
                    #print('Weight gradient')
                    #print(param.grad)
                    param -= learning_rate * param.grad
                    #print(param)

    def one_epoch_mini_batch_gd(self, train_data, i_model, size, cg_iterations=None,
                                learning_rate=0.005):
        #count = 0
        #print('Length of train_data: ', len(train_data))
        adam_optimizer = torch.optim.Adam(i_model.parameters())
        for data in train_data:
            '''
            if count % 10 == 0:
                print('Count: ', count)
            count = count + 1
            '''
            #print('Starting nodes: ', data[0], ' Ending nodes: ', data[1], ' Path lengths: ', data[2])
            grad = torch.zeros(size, 1, 1, dtype=torch.float64)
            index = torch.arange(len(data[0]))
            # Implementing mini-batch gradient descent
            loss = torch.DoubleTensor(1, 1)
            loss[0] = 0
            #print('Index: ', index)
            for i in index:
                #print(i)
                #i_model.zero_grad()
                adam_optimizer.zero_grad()
                # Continue implementing the Adam optimizer
                if cg_iterations is None:
                    h_pred = i_model(data[0][i], data[1][i])
                else:
                    h_pred = i_model(data[0][i], data[1][i], full=cg_iterations)
                a = torch.DoubleTensor(1)
                a[0] = data[2][i]
                # Original loss function
                loss += (1.0/len(index))*(a[0] - h_pred).pow(2)
                #loss += weights_matrix[data[0][i], data[1][i]]*(a[0] - h_pred).pow(2)
                #loss.backward()

                '''
                with torch.no_grad():
                    for param in i_model.parameters():
                        #print(param.grad)
                        grad += param.grad
                '''

            loss.backward()

            adam_optimizer.step()
            '''
            with torch.no_grad():
                for param in i_model.parameters():
                    param -= learning_rate * (grad/len(data))
                    # Add in a clause such that no weight vector value can below 0
                    param[param < 0.0] = 0.0
            '''

    def __call__(self, *args, **kwargs):
        f_params, w_vec = self.read_res_file_paths(self.res_filenames, res_dir=self.res_directory)
        smg = SyntheticMovGenerator(num_timesteps=self.num_timesteps, num_animals=self.num_animals,
                                         res_matrix=w_vec, matrix_params=f_params,
                                         filenames=self.mov_filenames, file_dir=self.file_dir, seed=self.seed)
        animal_movement_data, animal_names, max_trap = smg()
        #print('Sparse movement data')
        #print(animal_movement_data)

        train, test, validation = self.transform_dataset(animal_movement_data, max_trap, self.cam_traps,
                                                         self.batch_size, self.ratio)

        #print('True Conductance Matrix:')
        #print(self.compute_res_matrix(w_vec, f_params))
        w_vec_starting = np.load(os.path.normpath(os.path.join(self.res_directory, self.starting_vec[0])))

        if self.grad_type == 'laplacian':
            #model = InvLearnDirectLaplacian(f_params.shape[1], cond_matrix=None, base_matrices=f_params)
            model = InvLearnDirectLaplacian(f_params.shape[1], cond_matrix=None, base_matrices=f_params, weight_vector=w_vec_starting)
        elif self.grad_type == 'cg':
            model = InvLearnConjugateGradient(f_params.shape[1], cond_matrix=None, base_matrices=f_params, weight_vector=w_vec_starting)

        cond_matrix_true = self.compute_res_matrix(w_vec, f_params)
        best_matrix = []
        model_truth = InvLearnDirectLaplacian(f_params.shape[1], cond_matrix=cond_matrix_true)

        save_file = None
        loss_dict = None
        best_matrix_file = None
        if self.csv_file is not None:
            save_file = os.path.normpath(os.path.join(self.res_directory, self.csv_file + '.csv'))
            best_matrix_file = os.path.normpath(os.path.join(self.res_directory, self.csv_file + '_best_matrix'))
            loss_dict = {'Epochs': [], 'Training Loss': [], 'Testing Loss': [], 'Time': [], 'KL': []}

        #weights_matrix = self.weights_on_dataset(validation, cond_matrix_true)

        print('Benchmark Values:')
        self.compute_test_train_loss(validation, test, model_truth, -1, loss_dict, 0, None,
                                     cg_iterations=None, save_file=save_file,
                                     best_matrix_file=best_matrix_file)

        print('')
        print('Initial Loss Values')
        self.compute_test_train_loss(validation, test, model, 0, loss_dict, 0, best_matrix,
                                     cg_iterations=self.cg_iterations, save_file=save_file,
                                     best_matrix_file=best_matrix_file)
        self.compute_KL_divergence(loss_dict, best_matrix, w_vec, f_params, save_file=save_file)
        print('')

        #print('Initial Conductance Matrix:')
        #print(self.compute_res_matrix(w_vec_starting, f_params))
        #print('')

        print('Training Process: ')
        print('')
        print('Learning Rate: ', self.lr)

        for epoch in range(self.epochs):
            print('Epoch: ', epoch+1)
            start_time = time.perf_counter()
            #start_time = time.process_time()
            if self.batch_size == 1:
                self.one_epoch_sgd(train, model, learning_rate=self.lr, cg_iterations=self.cg_iterations)
            else:
                self.one_epoch_mini_batch_gd(train, model, len(w_vec), learning_rate=self.lr,
                                             cg_iterations=self.cg_iterations)
            #end_time = time.process_time()
            end_time = time.perf_counter()
            print('Epoch Time: ', end_time-start_time)
            print('')
            self.compute_test_train_loss(validation, test, model, epoch+1, loss_dict, end_time - start_time, best_matrix,
                                         cg_iterations=self.cg_iterations, save_file=save_file,
                                         best_matrix_file=best_matrix_file)
            self.compute_KL_divergence(loss_dict, best_matrix, w_vec, f_params, save_file=save_file)
            #print('Current Conductance Matrix:')
            #print(self.compute_res_matrix(model.print_weights_vector().detach().numpy(), f_params))
            if epoch == (self.epochs - 1):
                print(model.print_weights_vector())

        if save_file is not None:
            df = pd.DataFrame(data=loss_dict)
            f = open(save_file, 'w')
            cmd_args = str(self.res_filenames) + ' ' + str(self.starting_vec) + ' ' + str(self.res_directory) + \
                       ' ' + str(self.num_timesteps) + ' ' + str(self.num_animals) + ' ' + str(self.cam_traps) + ' ' + \
                       str(self.grad_type) + ' ' + str(self.epochs) + ' ' + str(self.batch_size) + ' ' + str(self.ratio)\
                       + ' ' + str(self.lr) + ' ' + str(self.csv_file) + ' ' + str(self.cg_iterations) + '\n'
            f.write(cmd_args)
            df.to_csv(f)
            np.save(best_matrix_file, best_matrix)


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Execute inverse learning problem using either direct matrix inverse '
                                                 'gradient or conjugate gradient method.')
    parser.add_argument('res_filenames', help="List of file locations for resistance factor parameters and associated "
                                              "weight vector. First filename indicates factor parameters. Second "
                                              "filename indicates weight vector. Third filename is the starting weight "
                                              "vector in training.", default=None, const=None, nargs=2)
    parser.add_argument('starting_weights', help="File name for weight vector to initialize the machine learning "
                                                 "training process.", nargs=1)
    parser.add_argument('res_directory', help='Indicate file directory of resistance factor parameters and weight '
                                              'vector. If not needed, specify N/A.', default='N/A')
    parser.add_argument("num_timesteps", help="Number of timesteps to synthetically generate animal movement data for "
                                              "each animal.", type=int)
    parser.add_argument("num_animals", help="Number of animals for which data should be generated for.", type=int)
    parser.add_argument('sparse_traps', help='Number of camera traps from which animal individual data can be collected'
                                             ' from. This must be less than the dimensions of the resistance matrix. '
                                             'An error will be thrown if this property is violated.', type=int)
    parser.add_argument('gradient_type', help="Type in either 'laplacian' for inverse learning using the direct matrix "
                                              "gradient or 'cg' for conjugate gradient. Not correctly specifying this argument will"
                                              "thrown an exception.", default=None)
    parser.add_argument('epochs', help="The number of epochs to conduct stochastic gradient descent on the training "
                                       "dataset.", type=int, default=1)
    parser.add_argument('batch_size', help="The size of each batch in mini-batch gradient descent.", type=int)
    parser.add_argument('--mov_filenames', help="List of filenames to store animal movement data. List one filename for"
                                            "storing animal movement data in CSV format and another filenames for "
                                            "storing auxiliary data.", dest="mov_files", default=None, const=None,
                        nargs='*')
    parser.add_argument('--file_dir', help="File directory to store animal movement data",
                        dest="dir", default=None, const=None, nargs='?')
    parser.add_argument('--ratio', help="The ratio to specify the amount of animal movement data to be segmented for a"
                                        " validation dataset. The default value is 0.0 signifying all the data is used "
                                        "for training. If there is no validation dataset, the test loss will be "
                                        "calculated on the entire dataset.", dest='ratio', default=0.0, nargs='?',
                        type=float, action='store')
    parser.add_argument('--learning_rate', help="The learning rate of stochastic gradient descent. If argument is not "
                                                "specified, the default learning rate is 0.005.", type=float, default=0.005,
                        nargs='?', dest='lr')
    parser.add_argument('--csv_file', help="Name of output file of training and testing loss for every epoch. Header "
                                           "contains command line argument to generate experiment. Will be saved in "
                                           "res_directory.", dest='cf')
    parser.add_argument('--num_iterations_cg', help='Number of iterations to run conjugate gradient', type=int,
                        nargs='?', dest='iterations', default=None)
    parser.add_argument("--seed", help="Seed of random number generator, automatically defaults to 42.", action='store',
                        default=42, dest='seed', nargs='?', const=42)
    args = parser.parse_args()
    if args.gradient_type != 'laplacian' and args.gradient_type != 'cg':
        raise Exception("Not a valid gradient type. Please specify either 'laplacian' or 'cg' gradient methods.")

    inv_l = InverseLearning(args.res_filenames, args.res_directory, args.num_timesteps, args.num_animals, args.sparse_traps, args.gradient_type, args.epochs,
                            args.batch_size, starting_vec=args.starting_weights, mov_filenames=args.mov_files,
                            file_dir=args.dir, ratio=args.ratio, learning_rate=args.lr, seed=args.seed, csv_file=args.cf,
                            cg_iterations=args.iterations)
    inv_l()

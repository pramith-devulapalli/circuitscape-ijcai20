import pandas as pd
import os
import numpy as np
#from InvLearnDataGenerator import MovementDataReader


class DatasetTransformer:
    def __init__(self, animal_mov_dataframes=None, max_cam_trap=None, ratio=0.0, filenames=None, file_dir=None,
                 num_cam_traps=None):
        self.animal_mov_dataframes = animal_mov_dataframes
        self.max_cam_trap = max_cam_trap
        self.num_cam_traps = num_cam_traps
        self.filenames = filenames
        self.ratio = ratio
        self.file_dir = file_dir

    def save_dataframe(self, dataframe, filename, file_dir=None):
        file_path = ''
        if self.file_dir is not None:
            file_path = os.path.normpath(os.path.join(file_dir, filename + '.csv'))
        else:
            file_path = os.path.normpath(filename + '.csv')
        dataframe.to_csv(file_path, index=False)

    def extract_all_hitting_times(self, animal_mov_dataframes, max_cam_trap):
        hitting_times_list = []
        for mov_dataframe in animal_mov_dataframes:
            hitting_times_list.append(pd.DataFrame(data=self.all_hitting_times(mov_dataframe, max_cam_trap)))

        return pd.concat(hitting_times_list, ignore_index=True, sort=False)

    def all_hitting_times(self, mov_dataframe, max_cam_trap):
        df_list = []
        for i in range(max_cam_trap):
            for j in range(max_cam_trap):
                if i != j:
                    self.single_hitting_time(mov_dataframe, i, j, df_list)
        return df_list

    def single_hitting_time(self, df, start, end, df_list):
        movement = df['Camera Trap']
        timestamps = df['Time']
        indicator = False
        temp_walk = []
        i = 0

        for loc in movement:
            if loc == start and indicator is False:
                temp_walk.append(timestamps.iloc[i])
                indicator = True
            elif loc == end and indicator is True:
                temp_walk.append(timestamps.iloc[i])
                indicator = False
                temp_dict = {'First Node': start, 'Last Node': end, 'Path Length': float(temp_walk[1] - temp_walk[0])}
                df_list.append(temp_dict)
                temp_walk = []
            i = i + 1
    '''
    def extract_all_hitting_times(self, animal_mov_dataframes, max_cam_trap):
        hitting_times_list = []
        for mov_dataframe in animal_mov_dataframes:
            hitting_times_list.append(self.one_animal_hitting_time(mov_dataframe, max_cam_trap))

        return pd.concat(hitting_times_list, ignore_index=True, sort=False)
    '''

    def one_animal_hitting_time(self, mov_dataframe, max_cam_trap):
        df_list = []
        dataframe = mov_dataframe['Camera Trap']
        timestamps = mov_dataframe['Time']
        #print('Time')
        print('One Animal Camera Trap Column')
        print(dataframe)
        cam_trap_arr = np.zeros(max_cam_trap)
        cam_matrix = np.zeros((max_cam_trap, max_cam_trap))
        print('Initialization')
        print(cam_trap_arr)
        i = 0
        prev_mov = 0
        for mov in dataframe:
            if i == 0:
                prev_mov = mov
                print('Step: ', i)
                print('Movement Step: ', mov)
                cam_trap_arr[mov] = timestamps.iloc[i]
                #cam_matrix[mov, mov] = timestamps.iloc[i]
            else:
                print('Step: ', i)
                print('Movement Step: ', mov)
                if prev_mov != mov:
                    cam_trap_arr[mov] = timestamps.iloc[i]
                    print('Camera trap array:')
                    print(cam_trap_arr)
                    self.extract_path_lengths(cam_trap_arr, mov, df_list, cam_matrix)
                else:
                    cam_trap_arr[mov] = timestamps.iloc[i]
                    print('Camera trap array:')
                    print(cam_trap_arr)
                prev_mov = mov
            i += 1
            #print(df_list)

        return pd.DataFrame(data=df_list)

    def extract_path_lengths(self, arr, idx, df_list, cam_matrix):
        first_nodes = []
        last_nodes = []
        lengths = []
        for element in range(len(arr)):
            if arr[int(element)] != 0 and element != idx:
                if cam_matrix[int(element), idx] != arr[int(element)]:
                    temp_dict = {'First Node': element, 'Last Node': idx, 'Path Length': arr[int(idx)] - arr[int(element)]}
                    print(temp_dict)
                    #df_list.append({'First Node': element, 'Last Node': idx, 'Path Length': arr[int(idx)] - arr[int(element)]})
                    df_list.append(temp_dict)
                    cam_matrix[int(element), idx] = arr[int(element)]
        #print(df_list)
        print(cam_matrix)

    def create_partial_trajectories(self, df, num_cam_traps, max_trap):
        sparse_traps = np.random.choice(max_trap, num_cam_traps, replace=False)
        #print('Sparse Traps: ')
        #print(sparse_traps)
        #print('Current Hitting Dataframe: ')
        #print(df)
        last_node_df = df[df['First Node'].isin(sparse_traps)]
        sparse_hitting_dataframe = last_node_df[last_node_df['Last Node'].isin(sparse_traps)]
        #print('Sparse Hitting Dataframe: ')
        #print(sparse_hitting_dataframe)
        return sparse_hitting_dataframe

    def split_train_test(self, df, ratio, num_cam_traps, max_trap, seed=42):
        np.random.seed(seed)
        nRows = int(len(df)*ratio)
        df_shuffled = df.sample(frac=1)
        test_df = df_shuffled.iloc[:nRows].reset_index(drop=True)
        #print('Test Df: ')
        #print(test_df)
        train_df = self.create_partial_trajectories(df_shuffled.iloc[nRows:].reset_index(drop=True), num_cam_traps,
                                                    max_trap)
        return train_df, test_df

    def __call__(self, *args, **kwargs):
        result = []
        hitting_times_dataframe = self.extract_all_hitting_times(self.animal_mov_dataframes, self.max_cam_trap)

        if self.ratio != 0.0:
            train_df, test_df = self.split_train_test(hitting_times_dataframe, self.ratio, self.num_cam_traps,
                                                      self.max_cam_trap)
            result.append(train_df)
            result.append(test_df)
        else:
            result.append(hitting_times_dataframe)

        if self.filenames is not None:
            if len(self.filenames) == len(result):
                iterate = zip(self.filenames, result)
                for x, y in iterate:
                    self.save_dataframe(x, y, file_dir=self.file_dir)
            else:
                raise Exception('Number of filenames does not match number of dataframes to save.')

        return result


#mdr = MovementDataReader.MovementDataReader(mov_file='dataframes_1', file_dir='../data')
#data = mdr()
#dt = DatasetTransformer(animal_mov_dataframes=data[0], max_cam_trap=data[2], ratio=0.2)
#print(dt())
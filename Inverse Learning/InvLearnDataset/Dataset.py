from torch.utils import data
import os
import pandas as pd
#from InverseLearning.InvLearnDataset.DatasetTransformer import DatasetTransformer
#from InverseLearning.InvLearnDataGenerator.MovementDataReader import MovementDataReader


class Dataset(data.Dataset):
    def __init__(self, hitting_time_data=None, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
                 file_dir=None):
        csv_path = ''
        if file_dir is not None and isinstance(hitting_time_data, str):
            csv_path = os.path.normpath(os.path.join(file_dir, hitting_time_data + '.csv'))
        elif isinstance(hitting_time_data, str):
            csv_path = os.path.normpath(hitting_time_data + '.csv')

        if csv_path != '':
            self.dataframe = pd.read_csv(csv_path, header=0)
        else:
            self.dataframe = hitting_time_data

        self.params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'drop_last': drop_last}
        self.animal_csv_file = csv_path
        self.file_dir = file_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        first_cam_trap = int(self.dataframe.iloc[idx]['First Node'])
        last_cam_trap = int(self.dataframe.iloc[idx]['Last Node'])
        path_length = self.dataframe.iloc[idx]['Path Length']
        return first_cam_trap, last_cam_trap, path_length

    def return_iterator(self):
        dataloader = data.DataLoader(self, **self.params)
        return dataloader

    def get_animal_csv_file(self):
        return self.animal_csv_file

    def get_file_dir(self):
        return self.file_dir

'''
mdr = MovementDataReader(mov_file='dataframes_1', file_dir='../data')
data_1 = mdr()
#print(data_1)
dt = DatasetTransformer(animal_mov_dataframes=data_1[0], max_cam_trap=data_1[2], ratio=0.1)
#print(dt())
ht_df = dt()[0]
dataset = Dataset(hitting_time_data=ht_df, batch_size=3)
print(len(dataset))
for i in range(len(dataset)):
    print(dataset[i])
data_loader = dataset.return_iterator()
print("Data Loader")
for row in data_loader:
    print(row)
'''
'''
print('Second Iteration')
for s_row in data_loader:
    print(s_row)

print(data_loader)
'''
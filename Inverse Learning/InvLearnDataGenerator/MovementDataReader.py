import sys
import argparse
import os
import pandas as pd
from dfply import *
import datetime


class MovementDataReader:
    def __init__(self, mov_file=None, file_dir=None):
        self.mov_file = mov_file
        self.file_dir = file_dir

    def read_csv_file(self, mov_file, file_dir=None):
        mov_file_path = ''
        if file_dir is not None:
            mov_file_path = os.path.normpath(os.path.join(file_dir, mov_file + '.csv'))
        else:
            mov_file_path = os.path.normpath(mov_file + '.csv')
        mov_data = pd.read_csv(mov_file_path, header=0)
        max_cam_trap = mov_data['Camera Trap'].max()
        animal_names = (mov_data >> select('Animal Individual') >> distinct).values.flatten().tolist()
        groupby_dataframes = mov_data.groupby('Animal Individual')
        animal_mov_dataframes = [groupby_dataframes.get_group(animal) for animal in animal_names]
        for df in animal_mov_dataframes:
            #list(df['Time']).sort(key=lambda date: datetime.datetime.strptime(date, '%m-%d-%y %H:%M:%S'))
            list(df['Time']).sort()
        return animal_mov_dataframes, animal_names, max_cam_trap

    def __call__(self, *args, **kwargs):
        mov_data = self.read_csv_file(self.mov_file, file_dir=self.file_dir)
        return mov_data


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Read in animal movement data file in specified CSV format')
    parser.add_argument("filename", help=".csv file to extract animal movement data, must contain relative or absolute "
                                         "path if file directory is not specified")
    parser.add_argument("directory", help="File directory where .csv file is located. Optional to indicate.", nargs='?',
                        default=None)
    args = parser.parse_args()
    mdr = MovementDataReader(mov_file=args.filename, file_dir=args.directory)
    print(mdr())


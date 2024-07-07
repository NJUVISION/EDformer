import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, random_split
import os

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3
LABEL_COLUMN = 4


class ED24(Dataset):
    def __init__(self, root, N=4096):
        voltages = ['1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2',
                    '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5']
        self.folders = os.listdir(root)
        self.data_files = [os.path.join(root, folder, file) for folder in self.folders for file in os.listdir(
            os.path.join(root, folder)) if (folder in file) and ('.'.join(file.split("_")[-1].split(".")[:2]) in voltages)]
        self.data_files.sort()
        self.N = N
        self.data = []
        for i, data_file in enumerate(tqdm.tqdm(self.data_files)):
            data = pd.read_csv(data_file, skiprows=0, delimiter=' ', dtype={
                               'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8})
            data = data.values
            if data.shape[0] < self.N:
                continue
            if (data.shape[0] % self.N) == 0:
                inputs = data
            else:
                inputs = data[:-(data.shape[0] % self.N), :]
            x = inputs[:, X_COLUMN]
            y = inputs[:, Y_COLUMN]
            polarity = inputs[:, POLARITY_COLUMN]
            timestamp = self.normalize_column(
                inputs[:, TIMESTAMP_COLUMN])
            label = inputs[:, LABEL_COLUMN]
            inputs = np.hstack((timestamp.reshape(-1, 1), x.reshape(-1, 1),
                               y.reshape(-1, 1), polarity.reshape(-1, 1), label.reshape(-1, 1)))
            M = inputs.shape[0] // self.N
            inputs = np.reshape(inputs, (M, self.N, 5))
            for input in inputs:
                input = torch.from_numpy(input).to(torch.float32)
                self.data.append(input)
            # if i > 20:
            #     break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.data[index]
        events = inputs[:, :4]
        labels = inputs[:, 4].unsqueeze(-1)
        return events, labels

    def normalize_column(self, column):
        min_val = np.min(column)
        max_val = np.max(column)
        normalized_column = (column - min_val) / (max_val - min_val)
        return normalized_column


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ED24 dataset')
    parser.add_argument('root', type=str, help='dataset root path')
    parser.add_argument('N', type=int, help='number of events')
    args = parser.parse_args()

    root = '/workspace/shared/event_dataset/ECCV2024_datasets/ED24'
    dataset = ED24(args.root, args.N)
    total_samples = len(dataset)
    print(total_samples)

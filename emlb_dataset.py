import os
import os.path as osp
from glob import glob
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, filepath):
        self.files = []
        for dirname in os.listdir(filepath):
            if os.path.isdir(osp.join(filepath, dirname)):
                for filename in os.listdir(osp.join(filepath, dirname)):
                    if filename.endswith('.aedat4'):
                        self.files.append((filepath, dirname, filename))
        self.files.sort()

    def __getitem__(self, index):
        fpath, fclass, fname = self.files[index]
        return fpath, fclass, fname

    def __len__(self):
        return len(self.files)


def Dataset(file_path):
    folders = [f for f in os.scandir(file_path) if f.is_dir() and not f.name.endswith('.zip')]
    return [BaseDataset(f.path) for f in folders]


if __name__ == '__main__':
    dataset = Dataset('/workspace/shared/event_dataset/emlb')
    for i, seq in enumerate(dataset):
        print(seq)
        break

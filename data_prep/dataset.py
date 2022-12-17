import torch
import json
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader

with open('../config.json', 'r') as f:
    config = json.load(f)

def get_pickle_from_name(name, processed_path):
    name_pickled = name.replace('.wav', '.pkl')
    path = processed_path + '/' + name_pickled
    return path


class Esc50Dataset(Dataset):
    def __init__(self, csv_file, processed_dir):
        """
        Args:
            csv_file (string): Path to csv file with labels.
            processed_dir (string): Path to directory with processed pickled spectrograms.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations['picklepath'] = self.annotations.apply(lambda x: get_pickle_from_name(x['filename'], processed_dir), axis=1)

    def __len__(self):
        return len(self.annotations)

    def __get_item__(self, idx):

        feature_info = self.annotations.iloc[idx]
        with open(feature_info.loc['picklepath'], 'rb') as f:
            spec = pickle.load(f)
        feature = {'path': feature_info.loc['picklepath'],
                   'data': spec,
                   'class': feature_info.loc['fold'],
                   'subclass': feature_info.loc['category']}
        return feature



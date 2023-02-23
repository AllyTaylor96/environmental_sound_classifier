import torch
import torchaudio
import json
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader

with open('config.json', 'r') as f:
    config = json.load(f)

def get_audio_path(name, processed_path):
    return processed_path + '/' + name


class Esc50Dataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        """
        Args:
            csv_file (string): Path to csv file with labels.
            audio_dir (string): Path to directory with raw audio.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations['audiopath'] = self.annotations.apply(lambda x: get_audio_path(x['filename'], audio_dir), axis=1)

    def __len__(self):
        return len(self.annotations)

    def __get_item__(self, idx):

        feature_info = self.annotations.iloc[idx]
        with open(feature_info.loc['audiopath'], 'rb') as f:
            audio_data = torchaudio.load(f)[0]
        feature = {'path': feature_info.loc['audiopath'],
                   'data': audio_data,
                   'class': feature_info.loc['fold'],
                   'subclass': feature_info.loc['category']}
        return feature

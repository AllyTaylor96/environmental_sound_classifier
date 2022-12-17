import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def label_race (row):
    return row.replace(

class Esc50Dataset(Dataset):
    def __init__(self, csv_file, processed_dir):
        """
        Args:
            csv_file (string): Path to csv file with labels.
            processed_dir (string): Path to directory with processed pickled spectrograms.
        """
        self.annotations = pd.read_csv(csv_file)
        print(self.annotations)


csv_file = '/home/ally_taylor/code/environmental_sound_classifier/data/labels/esc50.csv'
processed_dir = '/home/ally_taylor/code/environmental_sound_classifier/data/processed'
data = Esc50Dataset(csv_file, processed_dir)

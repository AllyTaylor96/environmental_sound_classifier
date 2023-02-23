import torch
import torchaudio
import pytorch_lightning as pl
import json
from data_prep.dataset import Esc50Dataset
from models.model import Wav2Vec2Base


with open('config.json', 'r') as f:
    config = json.load(f)

dataset = Esc50Dataset('./data/labels/esc50.csv', config['raw_audio_dir'])

test = dataset.__get_item__(5)
print(test['data'].shape)

model = Wav2Vec2Base

features, _ = model.extract_features(test['data'])

print(features.shape)



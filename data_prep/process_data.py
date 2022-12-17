"""Converts raw audio data into processed spectrograms."""

import torch
import torchaudio
import os
import pickle
import json

with open('../config.json', 'r') as f:
    config = json.load(f)

# instant spectrogram for data processing
spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=config['spec_n_fft'],
    win_length=config['spec_win_length'],
    hop_length=config['spec_hop_length'],
    center=True,
    normalized=True,
    pad_mode="reflect",
    power=2.0)

# process dataset into spectrograms
for filename in os.listdir(config['raw_audio_dir']):
    full_path = config['raw_audio_dir'] + '/' + filename
    waveform, sr = torchaudio.load(full_path)
    spec = spectrogram(waveform)
    spec_prepped = spec.squeeze()
    out_filename = config['processed_audio_dir'] + '/' + filename.replace('.wav', '.pkl')
    with open(out_filename, 'wb') as f:
        pickle.dump(spec_prepped, f)

print('Raw files processed and output to {}'.format(config['processed_audio_dir']))

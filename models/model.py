import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# use wav2vec 2.0 as base from torchaudio
bundle = torchaudio.pipelines.WAV2VEC2_BASE

Wav2Vec2Base = bundle.get_model()

# References: https://github.com/yxlu-0102/MP-SENet/blob/main/models/discriminator.py

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from models.lsigmoid import LearnableSigmoid1D
import numpy as np # ADD THIS LINE
import mir_eval.separation # ADD THIS LINE


# def pesq_loss(clean, noisy, sr=16000):
#     try:
#         pesq_score = pesq(sr, clean, noisy, 'wb')
#     except:
#         # error can happen due to silent period
#         pesq_score = -1
#     return pesq_score


# def batch_pesq(clean, noisy, cfg):
#     num_worker = cfg['env_setting']['num_workers']
#     pesq_score = Parallel(n_jobs=num_worker)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
#     pesq_score = np.array(pesq_score)
#     if -1 in pesq_score:
#         return None
#     pesq_score = (pesq_score - 1) / 3.5
#     return torch.FloatTensor(pesq_score)

def batch_sdr(clean_audios, enhanced_audios, sr=16000):
    """
    Calculates SDR for a batch of audio, used as target for discriminator.
    Args:
        clean_audios (np.ndarray): Clean waveforms (batch_size, samples).
        enhanced_audios (np.ndarray): Enhanced waveforms (batch_size, samples).
        sr (int): Sampling rate.
    Returns:
        torch.Tensor: Tensor of SDR scores for each sample in the batch.
    """
    sdr_scores = []
    for i in range(clean_audios.shape[0]): # Iterate through batch
        clean = clean_audios[i].flatten() # Ensure 1D
        enhanced = enhanced_audios[i].flatten() # Ensure 1D

        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
            np.expand_dims(clean, axis=0),
            np.expand_dims(enhanced, axis=0),
            compute_permutation=False
        )
        sdr_scores.append(sdr[0])
    return torch.tensor(sdr_scores, dtype=torch.float32)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid1D(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)

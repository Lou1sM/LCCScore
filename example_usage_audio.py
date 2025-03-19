from LCC.continuous_lcc import ContinuousLCCMeasurer
from scipy.signal import spectrogram
import librosa
import numpy as np
from PIL import Image


def resize(spec):
    new_size = 64*64
    spec_im = Image.fromarray(spec)
    h,w = spec_im.size[:2]
    aspect_ratio = h/w
    new_h = (new_size*aspect_ratio)**0.5
    new_w = new_size/new_h
    new_h_int = round(new_h)
    new_w_int = round(new_w)
    resized = spec_im.resize((new_h_int,new_w_int))
    resized = np.array(resized)
    resized = resized/resized.mean()
    resized = np.expand_dims(resized.T,1)
    return resized

example_audio_fp = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39586336.mp3'
audio, sr = librosa.load(example_audio_fp, offset=0.2, duration=1.0, mono=True) # takes a segment of length 1s, starting at 0.2s
comp_meas = ContinuousLCCMeasurer(ncs_to_check=8, n_levels=3, cluster_model='gmm')
sample_freqs, sample_points, spec = spectrogram(audio, fs=sr, nperseg=30)
resized = resize(spec)
model_costs_by_level, idx_costs_by_level, lccs_by_level, residuals_by_level, total_by_level = comp_meas.interpret(resized)
lcc_score = sum(lccs_by_level)
print(lcc_score)


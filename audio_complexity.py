import os
from io import BytesIO
import librosa
from skimage.measure import shannon_entropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LCC.continuous_lcc import ContinuousLCCMeasurer
from scipy.signal import spectrogram
from tqdm import tqdm
from PIL import Image
import argparse
from sections_of_audio import walla_fps, birdsong_fps_starts, birdsong_neg_fps_starts, orcavocs_fps_starts, orcavocs_neg_fps_starts, irish_s1_fps_starts, irish_s2_fps_starts, english_s1_fps_starts, english_s2_fps_starts, german_s1_fps_starts, german_s2_fps_starts, rain_fps_starts, tuning_forks_fps, bell_fps_starts
from baselines import im_compression_ratio


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=10)
parser.add_argument('--n-cluster-inits', type=int, default=1)
parser.add_argument('--exp', type=str, choices=['main', 'abl', 'each-level', 'sample-sizes', 'baselines'], default='main')
parser.add_argument('--show-spec-ims', action='store_true')
parser.add_argument('--normalize-results', action='store_true')
parser.add_argument('--is-test', '-t', action='store_true')
parser.add_argument('--verbose', action='store_true')
ARGS = parser.parse_args()

comp_meas = ContinuousLCCMeasurer(ncs_to_check=8,
                               n_cluster_inits=ARGS.n_cluster_inits,
                               nz=2,
                               n_levels=3,
                               cluster_model='gmm',
                               is_mdl_abl=False,
                               print_times=False,
                               display_cluster_label_imgs=False,
                               suppress_all_prints=True,
                               verbose=ARGS.verbose)


def resize(spec):
    scale_by = np.prod(spec.shape)/(16*740) # orig ims I used were 16*740, scale in prop to size
    new_size = 64*64*scale_by
    spec_im = Image.fromarray(spec)
    h,w = spec_im.size[:2]
    aspect_ratio = h/w
    new_h = (new_size*aspect_ratio)**0.5
    new_w = new_size/new_h
    new_h_int = round(new_h)
    new_w_int = round(new_w)
    max_possible_error = (new_h_int + new_w_int) / 2
    if not (new_h_int*new_w_int - new_size) < max_possible_error:
        breakpoint()
    resized = spec_im.resize((new_h_int,new_w_int))
    resized = np.array(resized)
    resized = resized/resized.mean()
    if ARGS.show_spec_ims:
        plt.imshow(resized[:,:100]); plt.show()
    resized = np.expand_dims(resized.T,1)
    #print('resized', resized.shape)
    return resized

def load_and_proc(audio_fp, offset=0, dur=None):
    fixed_sr = 44100
    if audio_fp=='white-noise':
        audio = np.random.rand(44100) - 0.5
    elif audio_fp=='gaussian-noise':
        audio = np.random.randn(44100)/4 # to make std same as for uniform
    else:
        audio, _ = librosa.load(audio_fp, sr=fixed_sr, offset=offset, duration=dur, mono=True)
    sample_freqs, sample_points, spec = spectrogram(audio, fs=fixed_sr, nperseg=30)
    resized = resize(spec)
    model_costs_by_level, idx_costs_by_level, lccs_by_level, residuals_by_level, total_by_level = comp_meas.interpret(resized)
    scores = {'LCCScore': sum(lccs_by_level)}
    if ARGS.exp=='main':
        scores['Residual Cost'] = sum(residuals_by_level)
        scores['Model Cost'] = sum(model_costs_by_level)
        scores['Idx Cost'] = sum(idx_costs_by_level)
    elif ARGS.exp=='baselines':
        scores['katz'] = katz_fracdim(audio, fixed_sr)
        scores['ent'] = shannon_entropy(resized)
        scores['zl comp ratio'] = im_compression_ratio(resized.squeeze(1), comp_format='png')
        scores['flac comp ratio'] = 0 if 'noise' in audio_fp else audio_compression_ratio(audio_fp, offset, dur)
    elif ARGS.exp == 'abl':
        scores['just one level'] = sum(comp_meas.interpret(resized, n_levels=1))
        scores['no mdl'] = sum(comp_meas.interpret(resized, is_mdl_abl=True))
    elif ARGS.exp == 'each-level':
        scores.update({f'level {i+1}':v for i,v in enumerate(lccs_by_level)})
    elif ARGS.exp == 'sample-sizes':
        scores[str(len(audio))] = scores.pop('ours')
        for start,end in [(18750,21250), (17500,22500), (15000,25000), (10000,30000), (5000,35000), (0,40000)]:
            size = end-start
            _, _, spec = spectrogram(audio[start:end], fs=fixed_sr, nperseg=30)
            resized = resize(spec)
            scores[size] = sum(comp_meas.interpret(resized))

    return scores

def katz_fracdim(wav, sr):
    xdist = 1/sr
    n = len(wav)
    dists = ((wav[1:] - wav[:-1])**2 + xdist)**0.5
    L = dists.sum()
    dists_from_start = ((wav[1:] - wav[0])**2 + (xdist*np.arange(len(wav)-1)**2)**0.5)
    d = dists_from_start.max()
    kscore = np.log(n) / (np.log(n) + np.log(d) - np.log(L))
    if kscore<0:
        breakpoint()
    return kscore

def comp_iter(fps_starts):
    avg_scores = {}
    all_scores = {}
    for i, (fp, start) in enumerate(pbar:=tqdm(fps_starts)):
        new_scores = load_and_proc(fp, start, dur=1.0)
        for k,v in new_scores.items():
            if i==0:
                all_scores[k] = [v]
                avg_scores[k] = v
            else:
                all_scores[k].append(v)
                avg_scores[k] = (v + avg_scores[k]*i)/(i+1)
        pbar.set_description(' '.join(f'{k}: {v:.3f}' for k,v in avg_scores.items()))
    return {k:np.array(v) for k,v in all_scores.items()}

def audio_compression_ratio(fp, offset, dur):
    import soundfile as sf
    file_sampled_audio, file_sr = librosa.load(fp, sr=None, offset=offset, duration=dur, mono=True)
    wav_stream = BytesIO()
    sf.write(wav_stream, file_sampled_audio, file_sr, format='wav', subtype='PCM_24')
    uncompressed_size = wav_stream.getbuffer().nbytes

    flac_stream = BytesIO()
    sf.write(flac_stream, file_sampled_audio, file_sr, format='flac', subtype='PCM_24')
    compressed_size = flac_stream.getbuffer().nbytes

    return uncompressed_size/compressed_size

all_results = {}
names_and_fpstarts = [
    ('bell', bell_fps_starts),
    ('white-noise',[('white-noise',0)]*ARGS.n),
    ('gaussian-noise',[('gaussian-noise',0)]*ARGS.n),
    ('walla', walla_fps),
    ('tuning-fork', tuning_forks_fps),
    ('birdsong', birdsong_fps_starts),
    ('birdsong-background', birdsong_neg_fps_starts),
    ('orcavocs', orcavocs_fps_starts),
    ('orcavocs-background', orcavocs_neg_fps_starts),
    ('irish-speaker-1', irish_s1_fps_starts),
    ('irish-speaker-2', irish_s2_fps_starts),
    ('english-speaker-1', english_s1_fps_starts),
    ('english-speaker-2', english_s2_fps_starts),
    ('german-speaker-1', german_s1_fps_starts),
    ('german-speaker-2', german_s2_fps_starts),
    ('rain', rain_fps_starts),
    ]

for signal_name, signal_fps_starts in names_and_fpstarts:
    print(signal_name)
    all_results[signal_name] = comp_iter(signal_fps_starts[:ARGS.n])

if ARGS.exp=='sample-sizes':
    global_max = max([x for v in all_results.values() for v1 in v.values() for x in v1])
    global_min = min([x for v in all_results.values() for v1 in v.values() for x in v1])
    maxs = {k:global_max for k in all_results['walla'].keys()}
    mins = {k:global_min for k in all_results['walla'].keys()}
else:
    maxs = {k:max(x for v in all_results.values() for x in v[k]) for k in all_results['walla'].keys()}
    mins = {k:min(x for v in all_results.values() for x in v[k]) for k in all_results['walla'].keys()}
scales = {k:100/(maxs[k]-mins[k]) for k in maxs.keys()}
mean_results = pd.DataFrame({k:{k1: x.mean() for k1,x in v.items()}  for k,v in all_results.items()})
std_results = pd.DataFrame({k:{k1: x.std()/10**0.5 for k1,x in v.items()}  for k,v in all_results.items()})
os.makedirs('results/audio', exist_ok=True)
mean_results.to_csv(f'results/audio/{ARGS.exp}-mean-results.csv')
std_results.to_csv(f'results/audio/{ARGS.exp}-std-results.csv')
if ARGS.normalize_results:
    df = pd.DataFrame({k:{k1:f'{(x.mean()-mins[k1])*scales[k1]:.1f} ({x.std()*scales[k1]/10**.5:.2f})' for k1,x in v.items()}  for k,v in all_results.items()})
else:
    df = pd.DataFrame({k:{k1:f'{(x.mean()):.1f} ({x.std()/10**.5:.2f})' for k1,x in v.items()}  for k,v in all_results.items()})
out_fpath = f'results/audio/{ARGS.exp}-results.tex'
if ARGS.n != 10:
    out_fpath = f'n{ARGS.n}-{out_fpath}'
if ARGS.exp=='sample-sizes':
    df.index=df.index.astype(int)
    df = df.sort_index()
    cs = {
        'walla': '#708090',
        'tuning-fork': '#708090',
        'birdsong': '#98FB98',
        'birdsong-background': '#B0C4DE',
        'orcavoc': '#2E8B57',
        'orcavoc-background': '#A9A9A9',
        'irish-s1-speech': '#FFA07A',
        'irish-s2-speech': '#FFA07A',
        'english-speech': '#DC143C',
        'english-s1-speech': '#DC143C',
        'english-s2-speech': '#DC143C',
        'german-s1-speech': '#FFA500',
        'german-s2-speech': '#FFA500',
        'rain': '#E6E6FA',
        }

    leg = {
        'walla': 'walla',
        'tuning-fork': 'tfork',
        'birdsong': 'bird',
        'birdsong-background': 'bird-bckgr',
        'orcavoc': 'orca',
        'orcavoc-background': 'orca-bckgr',
        'irish-s1-speech': 'ie-s1',
        'irish-s2-speech': 'ie-s2',
        'english-s1-speech': 'en-s1',
        'english-s2-speech': 'en-s2',
        'german-s1-speech': 'de-s1',
        'german-s2-speech': 'de-s2',
        'rain': 'rain',
        }
    for k in df.columns:
        means = []; stds = []
        for meanstd in df[k]:
            m, st = meanstd[:-1].split(' (')
            m = float(m); st = float(st)
            means.append(m); stds.append(st)
        means = np.array(means)
        stds = np.array(stds) / 10**0.5 / 2
        plt.plot(df.index, means, label=leg[k], color=cs[k])
        plt.fill_between(df.index, means-stds, means+stds, color=cs[k])
    #plt.legend(loc='upper left', handles=[leg[k] for k in df.columns])
    plt.legend(loc='upper left')
    plt.xticks(df.index, rotation=45)
    plt.xlabel('Num Samples')
    plt.ylabel('Relative Complexity Score')
    plt.tight_layout()
    plt.savefig('plot-vs-nsamples.png')
    os.system('/usr/bin/xdg-open plot-vs-nsamples.png')
print(df)

df.T.to_latex(out_fpath)

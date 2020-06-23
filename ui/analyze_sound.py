import argparse
import os
import json
import time
import sys

import librosa
import numpy as np
from sklearn.manifold import TSNE
import umap

def main():
    parser = argparse.ArgumentParser(description='generates a 2D mapping of timbre')
    parser.add_argument('file', metavar='FILE', help='the WAV file to be analyzed')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tsne', help='use t-SNE', action='store_true')
    group.add_argument('--umap', help='use UMAP', action='store_true')
    args = parser.parse_args()

    if args.tsne:
        process(args.file, method='tsne')
    elif args.umap:
        process(args.file, method='umap')
    else:
        assert False


def process(path, method='tsne'):
    directory = 'results'
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename, file_extension = os.path.splitext(path)
    split = os.path.split(filename)
    filename = split[1] + f'_{method}'
    result_path = os.path.join(directory, filename) + '.json'
    if os.path.exists(result_path):
       return result_path

    hop_length = 22050  # output length = (seconds) * (sample rate) / (hop_length)
    print(f'Snippet length = {hop_length / 22050:.2f}s')
    print('Loading WAV')
    y, sr = librosa.load(path)
    print('Computing MFCC features')
    mfcc = librosa.feature.mfcc(y, hop_length=hop_length, center=False)
    melspectrogram = librosa.feature.melspectrogram(y, hop_length=hop_length, center=False)
    rmse = librosa.feature.rms(y, hop_length=hop_length, center=False)
    spectral_centroid = librosa.feature.spectral_centroid(y, hop_length=hop_length, center=False)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y, hop_length=hop_length, center=False)
    spectral_contrast = librosa.feature.spectral_contrast(y, hop_length=hop_length, center=False)
    spectral_flattness = librosa.feature.spectral_flatness(y, hop_length=hop_length, center=False)
    spectral_rolloff = librosa.feature.spectral_rolloff(y, hop_length=hop_length, center=False)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, center=False)

    mfcc = mfcc.T
    n_hops = mfcc.shape[0]
    melspectrogram = melspectrogram.T
    rmse = np.squeeze(rmse)
    spectral_centroid = np.squeeze(spectral_centroid)
    spectral_bandwidth = np.squeeze(spectral_bandwidth)
    spectral_contrast = spectral_contrast.T
    spectral_flattness = np.squeeze(spectral_flattness)
    spectral_rolloff = np.squeeze(spectral_rolloff)
    zero_crossing_rate = np.squeeze(zero_crossing_rate)

    valid_indices = (rmse > 0.01)

    all_features = {
        'mfcc': mfcc[valid_indices],
        'rmse': rmse[valid_indices],
        'spectral': np.hstack(
            [spectral_centroid[valid_indices][:, np.newaxis], spectral_bandwidth[valid_indices][:, np.newaxis],
             spectral_contrast[valid_indices], spectral_flattness[valid_indices][:, np.newaxis],
             spectral_rolloff[valid_indices][:, np.newaxis], rmse[valid_indices][:, np.newaxis]])
    }
    all_features['all'] = np.hstack((all_features['mfcc'], all_features['spectral']))

    if method == 'tsne':
        print('Computing t-SNE')
        time_start = time.time()
        tsne = TSNE()
        Y = tsne.fit_transform(all_features['mfcc'])
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        result = {
            "x": Y[:, 0].tolist(),
            "y": Y[:, 1].tolist(),
            "t": ((np.arange(n_hops) * hop_length / 22050.0)[valid_indices]).tolist(),
            "filename": path
        }
        print(len(result['x']), 'elements')
    elif method == 'umap':
        print('Computing UMAP')
        time_start = time.time()
        Y = umap.UMAP(metric='correlation').fit_transform(all_features['mfcc'])
        result = {
            "x": Y[:, 0].tolist(),
            "y": Y[:, 1].tolist(),
            "t": ((np.arange(n_hops) * hop_length / 22050.0)[valid_indices]).tolist(),
            "filename": path
        }
        print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))

    with open(result_path, 'w') as f:
        json.dump(result, f)

    return result_path


if __name__ == '__main__':
    main()

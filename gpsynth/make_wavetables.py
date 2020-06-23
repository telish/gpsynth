import argparse
import datetime as dt
import os

from gpsynth.synthesizer import big_sweep, all_kernels

parser = argparse.ArgumentParser(description='Generate wavetables with Gaussian Processes')
parser.add_argument('path', metavar='path', type=str, nargs='?', default=None,
                    help='the parent directory, where the result is stored')
parser.add_argument('--lsdiv', metavar='N', type=int, required=False, default=16,
                    help='the number of lengthscale subdivisions')
parser.add_argument('--wavetables', metavar='N', type=int, required=False, default=7,
                    help='the number of (randomized) wavetables per setting of kernel and lengthscale')
args = parser.parse_args()

path = args.path
if path is None:
    dir_name = dt.datetime.now().strftime('%Y%m%d-%H%M') + '_multiexport'
    path = os.path.join(os.getcwd(), dir_name)

os.makedirs(path, exist_ok=True)
big_sweep(all_kernels, path, args.lsdiv, args.wavetables)

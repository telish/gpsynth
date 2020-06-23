"""
Call with --dir /path/to/DATE-TIME_multiexport
if you intend to use it with tsne_interface.py.
"""

import argparse
import os
import shutil
import json

from flask import Flask, render_template, request, send_from_directory
import hashlib
import analyze_sound

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = 'audio'
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        _, ext = os.path.splitext(file.filename)
        myfilename = md5(file)
        destination = os.path.join(target, myfilename) + ext
        file.save(destination)
        file.close()
        result_path = analyze_sound.process(destination, method='tsne')

    return render_template("visualization.html", json_file=result_path.replace('\\','/'))


@app.route('/audio/<path:path>')
def send_audio(path):
    return send_from_directory('audio', path)


@app.route('/results/<path:path>')
def send_result(path):
    return send_from_directory('results', path)


def md5(f):
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: f.read(4096), b''):
        hash_md5.update(chunk)
    f.seek(0)
    return hash_md5.hexdigest()


@app.route('/fixed')
def fixed():
    return render_template("visualization.html", json_file=fixed.result_path.replace('\\','/'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=False)
    args = parser.parse_args()
    if args.dir is not None:
        for file in os.listdir(args.dir):
            if file.endswith(".wav"):
                wav_path = os.path.join(args.dir, file)
                with open(wav_path, 'rb') as f:
                    target = 'audio'
                    if not os.path.exists(target):
                        os.mkdir(target)
                    myfilename = md5(f)
                    destination = os.path.join(target, myfilename) + '.wav'
                    shutil.copyfile(wav_path, destination)

                result_path = analyze_sound.process(destination, method='tsne')
                fixed.result_path = result_path
                print(f'Open http://127.0.0.1:4555/fixed')

                with open(os.path.join(args.dir, 'score.json'), 'r') as score_json:
                    score = json.load(score_json)
                    descriptions = []
                    for note in score:
                        waveshaping = 'waveshaping' if note['waveshaping'] else 'no waveshaping'
                        operator = '+' if note['operator'] == 'plus' else '*'
                        if note['kernel_2'] == '':
                            desc = f'{note["kernel_1"]}(l={note["lengthscale_1"]:.2f}), {waveshaping}'
                        else:
                            desc = f'{note["kernel_1"]}(l={note["lengthscale_1"]:.2f}) {operator} {note["kernel_2"]}' \
                                   f'(l={note["lengthscale_2"]:.2f}), {waveshaping}'
                        descriptions.append(desc)

                with open(fixed.result_path, 'r') as f:
                    results = json.load(f)
                    results['score'] = descriptions

                with open(fixed.result_path, 'w') as f:
                    json.dump(results, f)

                break

    app.run(port=4555)

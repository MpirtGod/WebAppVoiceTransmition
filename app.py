import io
import os
import uuid

import librosa
import torch
from flask import Flask, render_template, request, send_file

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# model = ModelInit()

app = Flask(__name__)

processor = Wav2Vec2Processor.from_pretrained("model/")
model = Wav2Vec2ForCTC.from_pretrained("model/")
device = torch.device('cpu')
model.to(device)

local_file = 'v3_1_ru.pt'
synt_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
synt_model.to(device)
sample_rate = 24000


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/project')
def project():
    return render_template("project.html")


def predict(speech_array, sampling_rate):
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    input_sequences = [processor.batch_decode(pred_ids)[0]]
    return input_sequences[0]


def synthesis(text, man=True):
    if man:
        speaker = 'aidar'
    else:
        speaker = 'baya'

    filename = str(uuid.uuid4()) + '.wav'
    synt_model.save_wav(text=text, speaker=speaker, sample_rate=sample_rate, audio_path=filename)
    return filename


@app.route('/upload', methods=['POST'])
def upload():
    gender = request.form.get('gender')
    file = request.files.get('file')

    if file:
        tmp = io.BytesIO(file.read())
        y, sr = librosa.load(tmp, sr=16000)
        text = predict(y, sr)
        filename = synthesis(text, gender == "man")
        with open(filename, 'rb') as f:
            file_content = f.read()
        file_stream = io.BytesIO(file_content)
        os.remove(filename)
        return send_file(file_stream, as_attachment=True, mimetype='blob', download_name="audio.wav")

    return "Данные отсутствуют или повреждены"


if __name__ == '__main__':
    app.run()

import io
import os
import uuid

import librosa
import torch
import torch.nn as nn
from flask import Flask, render_template, request, send_file
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class BiRNN(nn.Module):
    def __init__(self, rnn_dim, hidden_size):
        super(BiRNN, self).__init__()

        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.BiRNN = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiRNN(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechRecognitionModel, self).__init__()

        self.bundle = nn.Linear(20, 256)

        self.rnn1 = BiRNN(256, 256)

        self.rnn2 = BiRNN(256, 256)

        self.rnn3 = BiRNN(256, 256)

        self.rnn4 = BiRNN(256, 256)

        self.fc = nn.Linear(256, num_classes)

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.bundle(x)
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.rnn3(x)
        x = self.rnn4(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class ModelInit:
    def __init__(self, path='model.pth', device_type='cpu'):
        if device_type == 'cuda':
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda")
            else:
                print('Невозможно использовать GPU, выбран CPU')
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.model = SpeechRecognitionModel(35)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()

        self.char_map = {"а": 0, "б": 1, "в": 2, "г": 3, "д": 4, "е": 5, "ё": 6, "ж": 7, "з": 8, "и": 9, "й": 10,
                         "к": 11, "л": 12, "м": 13, "н": 14, "о": 15, "п": 16, "р": 17, "с": 18, "т": 19, "у": 20,
                         "ф": 21, "ч": 22, "ц": 23, "ш": 24, "щ": 25, "ъ": 26, "ы": 27, "ь": 28, "э": 29, "ю": 30,
                         "я": 31, "х": 32, " ": 33, "": 34}

        self.index_map = {}
        for key, value in self.char_map.items():
            self.index_map[value] = key

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string)

    def predict(self, audio):
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
        spectrogram_tensor = torch.FloatTensor(mfccs).squeeze(0).transpose(0, 1).unsqueeze(0)
        spectrogram_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(spectrogram_tensor)
            arg_maxes = torch.argmax(output, dim=2)
            decodes = []
            for i, args in enumerate(arg_maxes):
                decode = []
                for j, index in enumerate(args):
                    if index != 34:
                        if True and j != 0 and index == args[j - 1]:
                            continue
                        decode.append(index.item())
                decodes.append(self.int_to_text(decode))

        return decodes[0]


model = ModelInit()


app = Flask(__name__)

# processor = Wav2Vec2Processor.from_pretrained("model/")
# model = Wav2Vec2ForCTC.from_pretrained("model/")
# device = torch.device('cpu')
# model.to(device)

local_file = 'v3_1_ru.pt'
device = torch.device('cpu')
synt_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
synt_model.to(device)
sample_rate = 24000

ID_text = {}


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/project')
def project():
    return render_template("project.html")


# def predict(speech_array, sampling_rate):
#     inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(inputs.input_values.to(device)).logits
#     pred_ids = torch.argmax(logits, dim=-1)
#     input_sequences = [processor.batch_decode(pred_ids)[0]]
#     return input_sequences[0]


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
    ID = request.form.get('ID')
    print(ID)

    if file:
        tmp = io.BytesIO(file.read())
        y, sr = librosa.load(tmp, sr=16000)
        text = model.predict(y)
        ID_text[ID] = text
        filename = synthesis(text, gender == "man")
        with open(filename, 'rb') as f:
            file_content = f.read()
        file_stream = io.BytesIO(file_content)
        os.remove(filename)
        return send_file(file_stream, as_attachment=True, mimetype='blob', download_name="audio.wav")

    return "Данные отсутствуют или повреждены"


@app.route('/upload_text', methods=['POST'])
def upload_text():
    ID = request.form.get('ID')
    print(ID)

    if ID:
        try:
            text = ID_text[ID]
            del(ID_text[ID])
            return str(text)
        except:
            return "Отсутствие данных"

    return "Данные отсутствуют или повреждены"


if __name__ == '__main__':
    app.run()

import os
import io
import sys
import librosa
from flask import Flask, render_template, send_from_directory, request
from modelinit import ModelInit
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

root_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(root_dir, "templates")
js_dir = os.path.join(template_folder, 'js')
css_dir = os.path.join(template_folder, 'css')

model = ModelInit()

app = Flask(__name__, template_folder=template_folder)


# processor = Wav2Vec2Processor.from_pretrained("model/")
# model = Wav2Vec2ForCTC.from_pretrained("model/")
# device = torch.device('cpu')
# model.to(device)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory(js_dir, path)


@app.route("/css/<path:path>")
def send_css(path):
    return send_from_directory(css_dir, path)


# def predict(speech_array, sampling_rate):
#     inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(inputs.input_values.to(device)).logits
#     pred_ids = torch.argmax(logits, dim=-1)
#     input_sequences = [processor.batch_decode(pred_ids)[0]]
#     return input_sequences[0]


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    data = request.files.get('voice')
    if data:
        # file_content = data.read()
        # with open('audio.wav', 'wb') as f:
        #     f.write(file_content)
        tmp = io.BytesIO(data.read())
        y, sr = librosa.load(tmp, sr=16000)
        text = model.predict(y)
        # text = predict(y, sr)
        return text
    return "Данные отсутствуют или повреждены"
    # file = request.files['voice']
    # file_content = file.read()
    # with open('audio.wav', 'wb') as f:
    #     f.write(file_content)
    # return 'Audio uploaded successfully'


@app.route('/upload_wav', methods=['POST'])
def upload_wav():
    if 'file' not in request.files:
        return 'Файл отсутсвует'
    file = request.files['file']
    if file.filename == '':
        return 'Файл отсутсвует'
    tmp = io.BytesIO(file.read())
    y, sr = librosa.load(tmp, sr=16000)
    text = model.predict(y)
    return text


if __name__ == '__main__':
    app.run()

import os
from flask import Flask, render_template, send_from_directory, request

root_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(root_dir, "templates")
js_dir = os.path.join(template_folder, 'js')
css_dir = os.path.join(template_folder, 'css')

app = Flask(__name__, template_folder=template_folder)


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


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['voice']
    file_content = file.read()
    with open('audio.wav', 'wb') as f:
        f.write(file_content)
    return 'Audio uploaded successfully'


if __name__ == '__main__':
    app.run()

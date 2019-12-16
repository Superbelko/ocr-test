from flask import Flask, request, render_template, render_template_string, redirect, url_for
from werkzeug import secure_filename
import os 
import sys
import subprocess

UPLOAD_FOLDER = '/tmp'
TEMPLATE = r"""<!doctype html>
<title>OCR File</title>
<h1>OCR File</h1>
<form action='' method="POST" enctype="multipart/form-data">
    <p><input type='file' name='file'>
    <p><input type='text' name='width' placeholder='width'>
    <p><input type='text' name='height' placeholder='height'>
    <input type='submit' value='upload'>
    </p>
</form>
"""


# set windows temp dir path
if sys.platform == 'win32':
    UPLOAD_FOLDER = '.\\tmp'
    try:
        os.mkdir(UPLOAD_FOLDER)
    except FileExistsError:
        pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return redirect(url_for('ocr'))

@app.route('/ocr', methods=['GET','POST'])
def ocr():
    if request.method == 'POST':
        file = request.files['file']
        w = request.form.get('width', 320, type=int)
        h = request.form.get('height', 320, type=int)
        w = int(w) or 320
        h = int(h) or 320
        if file:
            filename = secure_filename(file.filename)
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(final_path)
            args = [sys.executable, 'combined.py', '--input', final_path, '--width', str(w), '--height', str(h)]
            ocr = subprocess.run(args, stdout=subprocess.PIPE)
            print(ocr)
            response = app.response_class(
                response=ocr.stdout,
                status=200,
                mimetype='application/json'
            )
            return response
    return render_template_string(TEMPLATE)


if __name__ == '__main__':
    app.run(debug=True)

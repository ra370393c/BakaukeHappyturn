import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import torch



classes = ["ばかうけ", "ハッピーターン"]

image_size = 224

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




model = CnnClassifier()
model.load_state_dict(torch.load("./trained_model0917_2.pth", map_location=torch.device('cpu')))



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size,3))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            if result[0] > result[1]:
                predicted = 0
                pred_answer = "この写真に写っているのは " + classes[predicted] + " ですね"
            elif result[0] < result[1]:
                predicted = 1
                pred_answer = "この写真に写っているのは" + classes[predicted] + " ですね"
            else:
                pred_answer = "この写真に写っているのは未知の生物ですね"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)

    

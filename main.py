import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchmetrics.functional import accuracy
from torchvision import transforms
import pytorch_lightning as pl




classes = ["ばかうけ", "ハッピーターン"]

image_size = 224

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




#　ネットワークの定義
class CnnClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #追加
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(37632, 2)

    def forward(self, x):
        #追加
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 37632)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y, t, task='multiclass', num_classes=2), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y, t, task='multiclass', num_classes=2), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y, t, task='multiclass', num_classes=2), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

        

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


            # 画像の前処理
            transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 画像サイズをモデルに合わせる
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 平均と標準偏差で正規化
            ])
            
            #受け取った画像を読み込み、np形式に変換
            img = Image.open(filepath)
            #img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size,3))
            #img = image.img_to_array(img)
            #data = np.array([img])
            img = transform(img).unsqueeze(0)  # バッチ次元を追加
            model.eval()  # 推論モードに切り替え
            #変換したデータをモデルに渡して予測する
            ## 判別
            with torch.no_grad():
                outputs = loaded_model(data)
                _, predicted = torch.max(outputs, 1)
                
            # predictedはクラスのインデックスです
            class_index = predicted.item()

            # 判別結果をクラス名に変換
            predicted_class_name = classes[class_index]


            if predicted_class_name == "ばかうけ":
                predicted = 0
                pred_answer = "この写真に写っているのは " + classes[predicted] + " ですね"
            elif predicted_class_name == "ハッピーターン":
                predicted = 1
                pred_answer = "この写真に写っているのは" + classes[predicted] + " ですね"
            else:
                pred_answer = "この写真に写っているのは未知のたべものですね"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)

    

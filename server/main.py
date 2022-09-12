from flask import Flask, render_template, request
from flask_cors import CORS
import torch
import sys
import os
sys.path.append(os.getcwd())
import utils

dev_type = utils.get_device_type()
model = utils.load_model(dev_type)
labels = utils.get_labels()

app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    dev_type = utils.get_device_type()
    labels = utils.get_labels()
    img = utils.image_loader(request.files['file'], dev_type)
    out = torch.exp(model(img))
    out_prob = torch.max(out).item()/torch.sum(out).item()
    cl = labels[out.argmax(dim=1)]
    return f'{cl} ({out_prob*100:.0f}%)'

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)



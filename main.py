from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file
# from werkzeug.utils import secure_filename
import os
from tools.crop_license_plate import crop_license_image
from tools.draw_annotation import yolo_anotation
import matplotlib.pyplot as plt
import torch
# from tools.model import cVAE
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
# from util import CustomDataset, thai_char_to_number
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
# from util import CustomDataset, thai_char_to_number
import matplotlib.pyplot as plt
import torch
# from tools.model import cVAE
import numpy as np
import cv2
from PIL import Image
import random
from tools.cvae_implement_same_class import CVAE, generate, mapping
import time
# from 

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

device = torch.device("cuda")
model = CVAE(28*28*3, 40, class_size=48).to(device)
model.load_state_dict(torch.load("tools/weight/best_model_cvae.pt"))
model.eval()


app = Flask(__name__)
# This is the maximum file size that can be uploaded
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# app.config['UPLOAD_FOLDER'] = 'uploads/'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    print('Checking if file allowed...')
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    print('pattrnit')  # This prints when the route is accessed, not necessarily when a file is uploaded.
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, the browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            return redirect(url_for('display_image', filename=file.filename))
    return render_template('index.html')

@app.route('/test_button', methods=['POST'])
def test_button():
    # Check if the post request has the file part
    # print(request.files['file'])
    if 'file' not in request.files:
        return jsonify(success=False, message="No file part")
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message="No selected file")
    if file:
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Process the file here and save the result
        processed_file_path = process_file(filename)
        # print(process_file)

        return send_file(processed_file_path, as_attachment=True)
    return jsonify(success=False)

def process_file(filename):
    file_path = crop_license_image(filename, "tools/weight/license_plate.pt")
    # print(file_path)
    file_result = yolo_anotation(str(file_path), 'tools/weight/digit.pt')
    return file_result

@app.route('/display_image/<filename>')
def display_image(filename):
    # Ensure the file exists to prevent arbitrary file access
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.isfile(filepath):
        flash('File not found.')
        return redirect(url_for('upload_file'))
    
    return render_template('display_image.html', filename=filename)

@app.route('/')
def home():
    cache_buster = time.time()
    return render_template('index2.html', cache_buster=cache_buster)

@app.route('/change_page', methods=['POST'])
def change_page():
    # Logic to handle the button click and change the page
    return redirect(url_for('new_page'))

@app.route('/find_out_more')
def new_page():
    return render_template('find_more.html')

@app.route('/Grayscale')
def new_page2():
    return render_template('Grayscale.html')

@app.route('/game')
def game_route():
    return render_template('game.html')

@app.route('/submit', methods=['POST'])
def submit():
    aa = cv2.imread("data_save/license_plate/license_plate_save.jpg")
    aa_resized = cv2.resize(aa, (640, 640))

    with open("data_save/labels/license_plate_save.txt", 'r') as text_file:
        lines = text_file.readlines()
    if request.is_json:
        data = request.get_json()  
        # print(data) 
        history = [mapping[i] for i in data.values()]
        for idx, (line, cha) in enumerate(zip(lines,history)):
            if str(cha) == '-' :
                continue
            components = line.strip().split()
            if len(components) == 5 and components[0] != "2กรุงเทพมหานคร":
                # components[0] = 0
                class_label, x1, y1, x2, y2 = map(float, components)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x1>640 :
                    x1 = 640
                if x2>640 :
                    x2 = 640
                if y1>640 :
                    y1 = 640
                if y2>640 :
                    y2 = 640            
                crop_image = aa_resized[y1:y2, x1:x2]
                # print(f'crop_image_size = {crop_image.shape}')
                crop_image_pil = Image.fromarray(crop_image)
                transformed_image = transform(crop_image_pil)
                image_from_cvae = generate(int(cha), model, transformed_image.unsqueeze(0))
                image_from_cvae_resized = cv2.resize(image_from_cvae, (x2 - x1, y2 - y1))
                aa_resized[y1:y2, x1:x2] = image_from_cvae_resized

        # with open(f'experiment/same_class/generation/generate_test.txt', "w") as output_file:
        #     for line in lines:
        #         output_file.write(line)           

        cv2.imwrite(f'static/generation/generate_test.jpg', cv2.resize(aa_resized, (123,64)))

        return redirect(url_for('home'))
    else:
        return jsonify({"error": "Request must be JSON"}), 400
    


if __name__ == '__main__':
    app.run(debug=True, port=8080)

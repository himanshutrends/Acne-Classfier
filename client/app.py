import gradio as gr
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import PIL 
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sentence_builder(age, sex, skin_type, allergies, diet, file):
    print(age, sex, skin_type, allergies, diet)
    
    img_byte_arr = BytesIO()
    file.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    payload = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}

    response = requests.post('http://127.0.0.1:5000/predict', files=payload)
    data = response.json()
    
    data['age'] = age
    data['gender'] = sex
    data['skin_type'] = skin_type
    data['allergies'] = allergies
    data['diet'] = diet

    response = requests.post('http://127.0.0.1:5000/recommendation', json=data)
    data = response.json()

    content = data['choices'][0]['message']['content']
    
    return content

demo = gr.Interface(
    sentence_builder,
    [
        gr.Number(value=20, label="Age"),
        gr.Radio(["Male", "Female", "Other"], label="Gender", info="Your Gender"),
        gr.CheckboxGroup(["Oily", "Dry", "Normal"], label="Skin", info="Skin Type"), 
        gr.Dropdown(
            ["benzoyl peroxide", "salicylic acid", "Sun-exposure", "Itching", "Swelling", "Redness"], 
            multiselect=True, label="Allergies", 
            info="Tell us your allergies and symptoms"
        ),
        gr.CheckboxGroup(["Veg", "Non-Veg",], label="Diet", info="Select your diet preference"),
        gr.Image(type="pil", label="Face Image (with open eye)"),
    ],
    gr.HTML("html"),
)

if __name__ == "__main__":
    demo.launch()   
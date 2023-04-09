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

    
    prediction = data['prediction']
    prediction = np.array(prediction)

    labels = ["Low", "Moderate", "Severe"]

    output1 = {labels[i]: float(prediction[0][i]) for i in range(3)}
    output2 = {labels[i]: float(prediction[1][i]) for i in range(3)}
    output3 = {labels[i]: float(prediction[2][i]) for i in range(3)}


    data['age'] = age
    data['gender'] = sex
    data['skin_type'] = skin_type
    data['allergies'] = allergies
    data['diet'] = diet

    response = requests.post('http://127.0.0.1:5000/recommendation', json=data)
    data = response.json()


    content = data['choices'][0]['message']['content']
    
    return content, output1, output2, output3

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
    outputs=[gr.HTML("html"), gr.Label(num_top_classes=3, label="Acne Level"), gr.Label(num_top_classes=3, label="Acne Level"), gr.Label(num_top_classes=3, label="Acne Level")]
)

if __name__ == "__main__":
    demo.launch()   
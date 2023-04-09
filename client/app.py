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

with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Row():
        with gr.Column():
            age = gr.Number(value=20, label="Age")
            sex = gr.Radio(["Male", "Female", "Other"], label="Gender", info="Your Gender")
            skin_type = gr.CheckboxGroup(["Oily", "Dry", "Normal"], label="Skin", info="Skin Type")
            allergy = gr.Dropdown(
                ["benzoyl peroxide", "salicylic acid", "Sun-exposure", "Itching", "Swelling", "Redness"],
                multiselect=True, label="Allergies", 
                info="Tell us your allergies and symptoms"
            )
            diet = gr.CheckboxGroup(["Veg", "Non-Veg",], label="Diet", info="Select your diet preference")
            img = gr.Image(source="webcam", type="pil", label="Face Image (with open eye)")
            submit = gr.Button("Submit")
            
        with gr.Tab("Model:Severity Prediction"):
            chin = gr.Label(num_top_classes=3, label="Acne Level")
            fh = gr.Label(num_top_classes=3, label="Acne Level")
            lc = gr.Label(num_top_classes=3, label="Acne Level")
        with gr.Tab("Recommendation:Treatment Plan"):
            html_output = gr.HTML('Recommendation will be shown here')

    submit.click(sentence_builder, inputs=[age, sex, skin_type, allergy, diet, img], outputs=[html_output, chin, fh, lc])

demo.queue().launch(share=True, debug=True, show_api=False, inline=False)
import gradio as gr
import requests
import numpy as np
from PIL import Image



def predict(input_img_numpy_array):
    img = Image.fromarray(input_img_numpy_array)
    x = requests.post('http://127.0.0.1:5000/predict', files=img)
    print(x)
    return img



demo = gr.Interface(fn=predict, inputs=gr.Image(shape=(200, 200)) , outputs="image")

demo.launch()   
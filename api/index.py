from flask import Flask
from flask import send_from_directory
from flask import request
import json
import get_patches
import model
from PIL import Image
import numpy as np
import os
import shutil
import openai


app = Flask(__name__)

@app.route('/temp/<path:path>')
def send_report(path):
    return send_from_directory('temp', path)

def convert_image(file_path):
  img = Image.open(file_path)
  img_re = img.resize((128, 128))
  numpy_arr = np.asarray(img_re)
  return numpy_arr

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      def extract_patches(imageURL):
        patch_path = 'patches'
        dimension_dict = dict()
        face_dict = dict()
        image_dim = []
        try:
          dim, face, img = get_patches.extract_patches(imageURL, dimension_dict, face_dict, image_dim, patch_path)
          print ("extract patches pass")
          return dim, face, img
        except:
          print ("extract patches fail")
          return None, None, None
      
      BASE_DIR = os.path.dirname(os.path.abspath(__file__))
      image = request.files['file']  
      imagePath = os.path.join(BASE_DIR, f"temp/{image.filename}")
      image.save(imagePath)  

      imageUrl = f"http://localhost:5000/temp/{image.filename}"

      os.mkdir(os.path.join(BASE_DIR, 'patches'))
      
      dim, face, img = extract_patches(imageUrl)

      if dim is None and face is None and img is None:
        return json.dumps({"msg": "fail"})

      model_path = os.path.join(BASE_DIR, 'models/Acne_Classifyer_N_Resnet.h5')

      resnet_model = model.load_trained_model(model_path)
      imageFiles = [os.path.join(BASE_DIR, 'patches', f) for f in os.listdir(os.path.join(BASE_DIR, 'patches'))]
      
      images = []
      landmarks = []
      for f in imageFiles:
        if f.endswith('.jpg'):
          landmark = f.split('/')[-1].split('_')[-1]
          landmarks.append(landmark)
          images.append(convert_image(f))

      images = np.array(images)
      
      prediction = resnet_model.predict(images)
      prediction = prediction.tolist()
      shutil.rmtree(os.path.join(BASE_DIR, 'patches'))
      return json.dumps({"msg": "success", "prediction": prediction, "landmarks": landmarks})

@app.route('/recommendation', methods=['POST'])
def recommendation():
  openai.api_key = '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'
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Hello!"},
  ]
)

  pass

if __name__ == '__main__':
  app.run(debug=True)
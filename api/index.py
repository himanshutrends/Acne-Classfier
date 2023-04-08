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
from dotenv import load_dotenv
load_dotenv()

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
                dim, face, img = get_patches.extract_patches(
                    imageURL, dimension_dict, face_dict, image_dim, patch_path)
                print("extract patches pass")
                return dim, face, img
            except:
                print("extract patches fail")
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

        model_path = os.path.join(
            BASE_DIR, 'models/Acne_Classifyer_N_Resnet.h5')

        resnet_model = model.load_trained_model(model_path)
        imageFiles = [os.path.join(BASE_DIR, 'patches', f)
                      for f in os.listdir(os.path.join(BASE_DIR, 'patches'))]

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
    openai.api_key = os.getenv("OPENAI_API_KEY")
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()

        if json.get('msg') == "success":
            prediction = json.get('prediction')
            prediction = np.array(prediction)
            prediction = prediction.argmax(axis=1)

            landmarks = json.get('landmarks')

            output = {landmarks[i].split(
                '.')[0]: f"level{prediction[i]}" for i in range(len(landmarks))}

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "you are a medical AI assistant that assists patients with acne on face with getting rid of it, okay? don't give me tips just nod"},

                    {"role": "user", "content": "what do you know about acne severity levels like level 0, level 1 and level 2 ?"},

                    {"role": "system", "content": """Acne severity levels are used to classify the severity of acne, with level 0 being the mildest and level 4 being the most severe. However, the commonly used system usually consists of four levels, which are described as follows:
                    Level 0: This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    Level 1: This level is used to describe skin with minor acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    Level 2: This level is used to describe skin with moderate acne. The skin may have more blackheads and whiteheads, as well as a few visible pimples.
                    Level 3: This level is used to describe skin with severe acne. The skin may have many blackheads, whiteheads, and visible pimples, as well as some nodules and cysts.
                    It is worth noting that acne severity can vary greatly from person to person, and there may be other factors to consider when assessing the severity of acne, such as the presence of scarring or inflammation. Therefore, it is important to consult with a dermatologist or healthcare professional for an accurate diagnosis and personalized treatment plan."""},

                    {"role": "user", "content": "okay let me teach u something new"},

                    {"role": "system", "content": "Sure, I'm always open to learning something new! What would you like to teach me?"},

                    {"role": "user", "content": """so level 0 is This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    level 1 This level is used to describe skin with minor acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    level 2 This level is used to describe skin with moderate acne to severe acne. The skin may have few to many blackheads and whiteheads, as well as a few or many visible pimples, as well as some nodules and cysts. there does not exist any other level these 3 are the only levels you know about okay"""},

                    {"role": "system", "content": """I apologize for the error in my previous response. You are correct that the commonly used acne severity classification system consists of three levels, not four. Here is the correct information:
                    Level 0: This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    Level 1: This level is used to describe skin with mild acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    Level 2: This level is used to describe skin with moderate to severe acne. The skin may have many blackheads and whiteheads, as well as a few or many visible pimples, as well as some nodules and cysts.
                    Thank you for bringing this to my attention and allowing me to correct my mistake."""},

                    {"role": "user", "content": "okay now i will tell you some instructions and the responses with boudries to which you will adhere"},

                    {"role": "system", "content": "Sure, please go ahead and give me the instructions and boundaries to adhere to."},

                    {"role": "user", "content": """so we have divided a face into patches these are namely fh(forehead), lc(left cheek), rc(right cheek), ch(chin). Each time my system will analyze the face image extract the patches from it and will send them to you it is not necessary that each time you will be provided with all the 4 mentioned patches above some time the message may contain only 1 or 2 patches. I will also send the acne severity level associated with that patch like 
                    {'fh':'level0', 'lc':'level2', 'rc':'level2', 'chin':'level0'}
                    this json data means that the patch forehead has level0 acne and left cheek has level2 acne and so on. Now that you now what level of acne does the parts of a particular face has you will generate a week long custom made remedy and daily routine for the acne treatment. I can ask you to generate a general acne routine and remedy but i am providing you with this classified details so that the output you will generate will be more optimised and fine grade. okay that's it if you understand compelety what i just instructed to you then generate a sample response for me for the above example data. also next time i will not be sending such long detailed message just the json message that i sent above so you will not get confused and do the task as i told you."""},

                    {"role": "system", "content": """Thank you for the detailed instructions. Yes, I understand what you have instructed me to do. Here's a sample response for the given example data:
                    For the patch on the forehead (fh) with level 0 acne, I recommend cleansing the area twice a day with a gentle cleanser and applying a non-comedogenic moisturizer afterwards. It is also important to protect the skin from UV damage by wearing a broad-spectrum sunscreen with at least SPF 30.
                    For the patch on the left cheek (lc) and right cheek (rc) with level 2 acne, I recommend using a salicylic acid or benzoyl peroxide-based cleanser in the morning and a glycolic or lactic acid-based cleanser in the evening to help exfoliate the skin and unclog pores. It is important to avoid using harsh scrubs or exfoliants as they can irritate the skin and worsen acne. Applying a topical acne treatment containing benzoyl peroxide or retinoids can also help reduce inflammation and breakouts. In addition, it is important to avoid touching the face and to change pillowcases regularly to prevent the transfer of bacteria.
                    For the patch on the chin (ch) with level 0 acne, the same routine as for the forehead patch can be followed.
                    It is important to note that the severity of acne can vary from person to person and it is recommended to consult with a dermatologist for personalized treatment options. This is a general recommendation based on the severity levels provided."""},

                    {"role": "user", "content": "{'fh':'level1', 'lc':'level0', 'rc':'level2'}"},

                    {"role": "system", "content": """Sure, here's a customized routine for the given example:
                    For the patch on the forehead (fh) with level 1 acne, I recommend using a gentle cleanser twice a day to help unclog pores and reduce inflammation. Applying a topical acne treatment containing salicylic acid or benzoyl peroxide can also be helpful in reducing breakouts. In addition, it is important to avoid picking or squeezing any existing pimples as this can lead to scarring and further breakouts.
                    For the patch on the left cheek (lc) with level 0 acne, a simple routine of cleansing twice a day with a gentle cleanser and applying a non-comedogenic moisturizer afterwards is recommended. It is also important to protect the skin from UV damage by wearing a broad-spectrum sunscreen with at least SPF 30.
                    For the patch on the right cheek (rc) with level 2 acne, I recommend using a salicylic acid or benzoyl peroxide-based cleanser in the morning and a glycolic or lactic acid-based cleanser in the evening to help exfoliate the skin and unclog pores. Applying a topical acne treatment containing benzoyl peroxide or retinoids can also help reduce inflammation and breakouts. In addition, it is important to avoid touching the face and to change pillowcases regularly to prevent the transfer of bacteria.
                    It is important to maintain a healthy diet and lifestyle as well, as this can also affect the severity of acne. Drinking plenty of water, eating a balanced diet with plenty of fruits and vegetables, and getting enough sleep can all contribute to healthy skin. If the acne does not improve or worsens, it is recommended to consult with a dermatologist for personalized treatment options."""},

                    {"role": "user", "content": """Also i need you to generate answers in way that they don't sound like they are being generated by a chat bot, leave out any sentences like "Sure, here's a customized routine for the given example:" so you need to sound like the answers are being generated by a deep learning NLP model not by a chat bot. You don't need to greet to me in the response just straight out tell the answer don't greet to me like 'here is the routine for the example' or anything like that if you understand generate the response for this json data {'fh':'level1', 'lc':'level0', 'rc':'level2', 'chin':'level1'} following my latest instructions."""},
                    {"role": "system", "content": """For the patch on the forehead with level 1 acne, a gentle cleanser twice a day and topical acne treatment containing salicylic acid or benzoyl peroxide can help unclog pores and reduce inflammation. For the patch on the left cheek with level 0 acne, a simple routine of cleansing twice a day and applying a non-comedogenic moisturizer afterwards is recommended. For the patch on the right cheek with level 2 acne, using a salicylic acid or benzoyl peroxide-based cleanser in the morning and a glycolic or lactic acid-based cleanser in the evening, along with topical acne treatment containing benzoyl peroxide or retinoids can help reduce inflammation and breakouts. For the patch on the chin with level 1 acne, a gentle exfoliant can help remove dead skin cells and unclog pores, followed by a topical acne treatment containing salicylic acid or benzoyl peroxide. It is important to maintain a healthy diet and lifestyle and avoid picking or squeezing any existing pimples. If the acne does not improve or worsens, it is recommended to consult with a dermatologist for personalized treatment options."""},
                    {"role": "user", "content": f"{output}"},
                ]
            )

        return completion


if __name__ == '__main__':
    app.run(debug=True)

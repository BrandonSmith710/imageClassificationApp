from transformers import ViTFeatureExtractor, ViTForImageClassification
from flask import Flask, request, render_template
import requests, PIL, pickle, os, sys

app = Flask(__name__)

feature_extractor = pickle.load(
    open('feature_extractor.h5', 'rb')
)

# transormer predicts one of the 1000 ImageNet classes
transformer = pickle.load(
    open('transformer_model.h5', 'rb')
)

@app.route('/', methods = ['POST', 'GET'])
def root():
    '''enter the directory and subdirectories(if any) separated by comma
    in the top field, and enter the image filename to classify in the
    bottom field'''

    if request.method == 'POST':
        directories = request.form.get('directoryField').split(',')
        image_name = request.form.get('imageField')
        folder_path = 'C:\\Users\\18186'
        directories = [d.strip() for d in directories]
        folder_path = os.path.join(folder_path, '\\'.join(directories))
        try:
            dir_list = os.listdir(folder_path)
        except Exception as e:
            print(e)
            return render_template('base.html')
        
        if image_name in dir_list:
            for folder, subfolders, images in os.walk(folder_path):   
                for img in images:        
                    if img == image_name:
                        folder_path = os.path.join(folder_path, img)
                        image = PIL.Image.open(folder_path)
                        inputs = feature_extractor(images = image,
                        return_tensors = "pt")
                        outputs = transformer(**inputs)
                        logits = outputs.logits
                        predicted_class_idx = logits.argmax(-1).item()
                        res = transformer.config.id2label[predicted_class_idx]
                        return render_template('results.html', answer = res)
                return render_template('base.html')
        return render_template('base.html')
    return render_template('base.html')

@app.route('/classify_multiple', methods = ['POST', 'GET'])
def classify_multiple():
    if request.method == 'POST':
        folder_path = 'C:\\Users\\18186'
        added_path = request.form.get('addedPath').split(',')
        added_path = [d.strip() for d in added_path]
        folder_path = os.path.join(folder_path, '\\'.join(added_path))
        res, s, k = [], set(), dict()
        try:
            dir_list = os.listdir(folder_path)
        except Exception as e:
            print(e)
            return render_template('base2.html')
        for folder, subfolders, images in os.walk(folder_path):
            for img in images:
                folder_pathx = os.path.join(folder_path, img)
                try:
                    image = PIL.Image.open(folder_pathx)
                except Exception as e:
                    print(img + ':', e)
                    continue
                inputs = feature_extractor(images = image,
                return_tensors = "pt")
                outputs = transformer(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_class = transformer.config.id2label[predicted_class_idx]
                res.append((img, predicted_class))
                s.add(predicted_class)
                if not k.get(predicted_class, 0):
                    k[predicted_class] = 1
                else:
                    k[predicted_class] += 1
        result, l = sorted(s, key = lambda x: k[x]), len(res)
        result = str([f'{label}: {round(k[label] / l * 100, 2)}%' for label in result]) + (
        '*'*50) + str(res)
        return render_template('results2.html', answer = result)
    return render_template('base2.html')

@app.route('/paste_and_classify', methods = ['POST', 'GET'])
def paste_and_classify():
    if request.method == 'POST':
        image = request.form.get('pastedImage')
        pass
    pass




if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 8080, debug = True)

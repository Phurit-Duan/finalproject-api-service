from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from PIL import Image
from django.http import FileResponse
import io
from base64 import b64decode
import tensorflow as tf
import os
import numpy as np
import json
from business_logic.Objectdetection import detection
from langdetect import detect
import re
import string
import pickle
from pythainlp.tokenize import word_tokenize

# Create your views here.
def test(request):
    return JsonResponse({'Text': 'HI'})

def decode_predictions(preds, top=3, class_list_path='././model-pre.json'):
  if len(preds.shape) != 2 or preds.shape[1] != 5:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  index_list = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(index_list[str(i)]) + (np.round(100*(pred[i]),5),) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

@csrf_exempt
def process_image(request):
    result = {}
    model = tf.keras.models.load_model('././fo_v2.h5') 
    if request.method == 'POST':
        file_uploaded = request.FILES.get('image')
        image = Image.open(file_uploaded)
        image = tf.keras.preprocessing.image.array_to_img(image.resize((128, 128)))
        x = np.expand_dims(image, axis=0)
        x_input = tf.keras.applications.mobilenet_v2.preprocess_input(x, data_format=None)
        predictions = model.predict(x_input)
        result_pred = decode_predictions(predictions)
        for i in range(len(result_pred[0])):
            result[i] = [result_pred[0][i][0], result_pred[0][i][1], result_pred[0][i][2]]
        content_type = file_uploaded.content_type
        response = { "status" : "POST API and you have uploaded a {} file".format(content_type) , "result" : result }
    else:
        response = { "status" : "none" }
    return JsonResponse(response)

@csrf_exempt
def object_detection_api(api_request):
    json_object = {'success': False}

    if api_request.method == "POST":

        if api_request.POST.get("image64", None) is not None:
            base64_data = api_request.POST.get("image64", None).split(',', 1)[1]
            data = b64decode(base64_data)
            data = np.array(Image.open(io.BytesIO(data)))
            result, detection_time = detection(data)

        elif api_request.FILES.get("image", None) is not None:
            image_api_request = api_request.FILES["image"]
            image_bytes = image_api_request.read()
            image = np.array(Image.open(image_api_request))
            result, detection_time = detection(image, web=True)

    if result:
        json_object['success'] = True
    json_object['time'] = str(round(detection_time))+" seconds"
    json_object['objects'] = result
    print(json_object)
    return JsonResponse(json_object)

@csrf_exempt
def nlp(request):
    text = request.POST.get('text')
    text = re.sub(r'<.*?>','', text)  
    text = re.sub(r'#','',text)  
    text = re.sub(r'…','',text)   
    for c in string.punctuation:
        text = re.sub(r'\{}'.format(c),'',text)
    text = ' '.join(text.split())
    print('clean text : ',text)
    language = detect(text)
    if (language == 'th'):
        vocabulary = pickle.load(open('././nlp-vocabulary.pkl', 'rb'))
        NLP_model  = pickle.load(open('././nlp-model.pkl',  'rb'))
        featurized_test_sentence =  {i:(i in word_tokenize(text.lower())) for i in vocabulary} 
        response = {"test_sent":text ,"result":NLP_model.classify(featurized_test_sentence)}
    else:
        response = {"test_sent":text ,"result":"Sorry!! This language is not supported, please send a message in Thai."}
    return JsonResponse(response)

@csrf_exempt
def test_image(request):
    file_uploaded = request.FILES.get('file_uploaded')
    return HttpResponse(file_uploaded, content_type="image/jpg")


@csrf_exempt
def test_text(request):
    text = request.POST.get('text')
    response = { "result" : text }
    return JsonResponse(response)

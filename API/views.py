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
import re
import string
import pickle
from pythainlp.tokenize import word_tokenize
from detect import detection
from django.core.files.storage import FileSystemStorage
from langdetect import detect
import re
import string
import pickle
from pythainlp.tokenize import word_tokenize


def decode_predictions(preds, top=3, class_list_path='././bakery-classes.json'):
  index_list = json.load(open(class_list_path))
  if len(preds.shape) != 2 or preds.shape[1] != len(index_list):
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(index_list[str(i)]) + (np.round(100*(pred[i]),5),) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

@csrf_exempt
def bakery_process_image(request):
    result = {}
    model = tf.keras.models.load_model('././bakery.weights-35-0.87.hdf5') 
    if request.method == 'POST':
        file_uploaded = request.FILES.get('image')
        image = Image.open(file_uploaded)
        image = tf.keras.preprocessing.image.array_to_img(image.resize((224, 224)))
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
def amulet_process_image(request):
    result = {}
    model = tf.keras.models.load_model('././amulet-weights-45-0.18.hdf5') 
    if request.method == 'POST':
        file_uploaded = request.FILES.get('image')
        image = Image.open(file_uploaded)
        image = tf.keras.preprocessing.image.array_to_img(image.resize((224, 224)))
        x = np.expand_dims(image, axis=0)
        x_input = tf.keras.applications.mobilenet_v2.preprocess_input(x, data_format=None)
        predictions = model.predict(x_input)
        result_pred = decode_predictions(predictions, class_list_path='././amulet-classes.json')
        for i in range(len(result_pred[0])):
            result[i] = [result_pred[0][i][0], result_pred[0][i][1], result_pred[0][i][2]]
        content_type = file_uploaded.content_type
        response = { "status" : "POST API and you have uploaded a {} file".format(content_type) , "result" : result }
    else:
        response = { "status" : "none" }
    return JsonResponse(response)

@csrf_exempt
def thai_cash_process_image(request):
    result = {}
    if request.method == 'POST':
        file_uploaded = request.FILES.get('image')
        fs = FileSystemStorage()
        filename = fs.save(file_uploaded.name, file_uploaded)
        uploaded_file_url = fs.url(filename)
        result = detection(source=uploaded_file_url, weights='best.pt', imgsz=640)
        fs.delete(file_uploaded.name)
        content_type = file_uploaded.content_type
        response = { "status" : "POST API and you have uploaded a {} file".format(content_type) , "result" : result }
    else:
        response = { "status" : "none" }
    return JsonResponse(response)

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


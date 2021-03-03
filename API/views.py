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
<<<<<<< HEAD
import json 
=======
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import pickle
>>>>>>> 2094086a2fc281f866b6ae2e6734abde71106116

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
def nlp(request):
    with codecs.open('pos.txt', 'r', "utf-8") as f: # โค้ดสำหรับเรียกไฟล์
        lines = f.readlines()
    listpos=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listpos
    del lines
    f.close()

    with codecs.open('neg.txt', 'r', "utf-8") as f:
        lines = f.readlines()
    listneg=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneg
    f.close()


    pos1=['pos']*len(listpos) # # ทำข้อมูล listpos ในให้เป็น pos
    neg1=['neg']*len(listneg) # ทำข้อมูล listneg ให้เป็น neg
    training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1)) # เทรนข้อมูล(คำ)ใน listpos และ listneg ให้เป็น neg และ pos แล้วเก็บไว้ใน training_data


    vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data])) 
    feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data] 
    classifier = nbc.train(feature_set) 


    test_sentence = request.POST.get('text') 
    featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary} 
    response = {"test_sent":test_sentence 
                ,"tag":classifier.classify(featurized_test_sentence)} # ใช้โมเดลที่ train ประมวลผล
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


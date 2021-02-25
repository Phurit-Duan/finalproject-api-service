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

# Create your views here.
def test(request):
    return JsonResponse({'Text': 'HI'})

@csrf_exempt
def process_image(request):
    class_name = ['berry','bird','dog','flower','other']
    model = tf.keras.models.load_model('././fo.h5') 
    if request.method == 'POST':
        file_uploaded = request.FILES.get('image')
        image = Image.open(file_uploaded)
        image = tf.keras.preprocessing.image.array_to_img(image.resize((128, 128)))
        x = np.expand_dims(image, axis=0)
        x_input = tf.keras.applications.mobilenet_v2.preprocess_input(x, data_format=None)
        predictions = model.predict(x_input)
        result = class_name[(int(np.argmax(predictions[0])))]
        content_type = file_uploaded.content_type
        response = { "status" : "POST API and you have uploaded a {} file".format(content_type) , "result" : result }
    else:
        response = { "status" : "none" }
    return JsonResponse(response)

@csrf_exempt
def test_image(request):
    file_uploaded = request.FILES.get('file_uploaded')
    return HttpResponse(file_uploaded, content_type="image/jpg")
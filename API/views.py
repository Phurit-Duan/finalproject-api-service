from django.views.decorators.csrf import csrf_exempt
import io
import os
import re
import string
import pickle
from langdetect import detect
from pythainlp.tokenize import word_tokenize
from business_logic.process_image import process_image_prediction, process_image_detect
from business_logic.process_text import process_nlp_prediction

@csrf_exempt
def bakery_process_image(request):
    path_model = "././bakery.weights-35-0.87.hdf5"
    path_class = "././bakery-classes.json"
    response = process_image_prediction(request, path_model, path_class)
    return response


@csrf_exempt
def amulet_process_image(request):
    path_model = "././amulet-weights-45-0.18.hdf5"
    path_class = "././amulet-classes.json"
    response = process_image_prediction(request, path_model, path_class)
    return response


@csrf_exempt
def thai_cash_process_image(request):
    response = process_image_detect(request)
    return response


@csrf_exempt
def nlp(request):
    response = process_nlp_prediction(request)
    return response

import re
import string
import pickle
from langdetect import detect
from pythainlp.tokenize import word_tokenize
from django.http import JsonResponse



def process_nlp_prediction(request):
    text = request.POST.get("text")
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"…", "", text)
    for c in string.punctuation:
        text = re.sub(r"\{}".format(c), "", text)
    text = " ".join(text.split())
    language = detect(text)
    if language == "th":
        vocabulary = pickle.load(open("././nlp-vocabulary.pkl", "rb"))
        NLP_model = pickle.load(open("././nlp-model.pkl", "rb"))
        featurized_test_sentence = {
            i: (i in word_tokenize(text.lower())) for i in vocabulary
        }
        response = {
            "test_sent": text,
            "result": NLP_model.classify(featurized_test_sentence),
        }
    else:
        response = {
            "test_sent": text,
            "result": "Sorry!! This language is not supported, please send a message in Thai.",
        }
    return JsonResponse(response)
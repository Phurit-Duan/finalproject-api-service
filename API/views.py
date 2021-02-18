from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from PIL import Image


# Create your views here.
def test(request):
    return JsonResponse({'Text': 'HI'})

@csrf_exempt
def process_image(request):
    if request.method == 'POST':        
        file_uploaded = request.FILES.get('file_uploaded')
        content_type = file_uploaded.content_type
        img = Image.load_img(file_uploaded, target_size=(128, 128))
        x = Image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        response = "POST API and you have uploaded a {} file".format(content_type)
    else:
        response = "Hi"
    return HttpResponse(response)
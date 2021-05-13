import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from django.http import JsonResponse
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from django.core.files.storage import FileSystemStorage


def decode_predictions(preds, top=3, class_list_path="././bakery-classes.json"):
    index_list = json.load(open(class_list_path))
    if len(preds.shape) != 2 or preds.shape[1] != len(index_list):
        raise ValueError(
            "`decode_predictions` expects "
            "a batch of predictions "
            "(i.e. a 2D array of shape (samples, 1000)). "
            "Found array with shape: " + str(preds.shape)
        )
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [
            tuple(index_list[str(i)]) + (np.round(100 * (pred[i]), 5),)
            for i in top_indices
        ]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def process_image_prediction(request, model_path, class_path):
    result = {}
    model = load_model(model_path)
    if request.method == "POST":
        try:
            file_uploaded = request.FILES.get("image")
            image_file = Image.open(file_uploaded).convert("RGB")
            img_numpy_array = img_to_array(image_file)
            image = array_to_img(image_file.resize((224, 224)))
            x = np.expand_dims(image, axis=0)
            x_input = preprocess_input(x, data_format=None)
        except:
            response = {
                "status": "POST API and you have uploaded a {} file".format(
                    file_uploaded.content_type
                ),
                "message": "this is not a valid bitmap file, or its format is not currently supported",
            }
            return JsonResponse(response, status=400)
        predictions = model.predict(x_input)
        result_pred = decode_predictions(predictions, class_list_path=class_path)
        for i in range(len(result_pred[0])):
            result[i] = [
                result_pred[0][i][0],
                result_pred[0][i][1],
                result_pred[0][i][2],
            ]
        response = {
            "status": "POST API and you have uploaded a {} file".format(
                file_uploaded.content_type
            ),
            "result": result,
        }
        return JsonResponse(response)
    else:
        response = {"status": "Error"}
        return JsonResponse(response)


def process_image_detect(request):
    result = {}
    if request.method == "POST":
        try:
            file_uploaded = request.FILES.get("image")
            fs = FileSystemStorage()
            filename = fs.save(file_uploaded.name, file_uploaded)
            uploaded_file_url = fs.url(filename)
            result = detect(source=uploaded_file_url, weights="./best.pt", imgsz=640)
            fs.delete(file_uploaded.name)
        except:
            response = {
                "status": "POST API and you have uploaded a {} file".format(
                    file_uploaded.content_type
                ),
                "message": "this is not a valid bitmap file, or its format is not currently supported",
            }
            return JsonResponse(response, status=400)
        response = {
            "status": "POST API and you have uploaded a {} file".format(
                file_uploaded.content_type
            ),
            "result": result,
        }
        return JsonResponse(response)
    else:
        response = {"status": "Error"}
        return JsonResponse(response)


def detect(source="data/images", weights="best.pt", imgsz=640):
    result = []

    # Initialize
    set_logging()
    device = select_device("")
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment="store_true")[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, 0.40, 0.45, classes=None, agnostic="store_true"
        )
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}"  # add to string
                    result.append(s)

    return result
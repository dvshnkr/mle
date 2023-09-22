# pylint: disable = E0401, E1101, W0613,

from io import BytesIO
from typing import Tuple
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image


def download_image(url: str) -> Image.Image:
    """
    Downloads image.
    """

    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def tf_preprocessing(x: np.ndarray):
    """
    tf_preprocessing from TensorFlow
    """

    x /= 127.5
    x -= 1.0
    return x


def caffe_preprocessing(x: np.ndarray):
    """
    caffe_preprocessing from TensorFlow
    """

    # 'RGB'->'BGR'
    x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


def prepare_image(img: Image.Image, target_size: Tuple[int, int] = (224, 224)):
    """
    Prepares image.
    """

    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)

    x = np.array(img, dtype="float32") / 255.0

    return np.expand_dims(x, axis=0)


def img_from_url(url: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Returns prepared image from given url.
    """

    img = download_image(url)
    return prepare_image(img, target_size)


def predict(url: str) -> float:
    """
    Downloads image, predicts class of image.
    """

    target_size = (150, 150)

    interpreter = tflite.Interpreter(model_path="dino_dragon_10_0.899.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    img = img_from_url(url, target_size)

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    pred_proba = preds[0].tolist()

    return pred_proba[0]


def lambda_handler(event, context=None):
    """
    Handles lambda event on AWS.
    """
    url = event["url"]
    result = predict(url)

    return result

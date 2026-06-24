
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

def load_and_preprocess(img_path, target_size=IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_pre = preprocess_input(img_resized.astype("float32"))
    return np.expand_dims(img_pre, axis=0)

def preprocess_from_bytes(file_bytes, target_size=IMG_SIZE):
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_pre = preprocess_input(img_resized.astype("float32"))
    return np.expand_dims(img_pre, axis=0)

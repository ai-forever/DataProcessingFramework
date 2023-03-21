from io import BytesIO
import numpy as np
from PIL import Image


def default_preprocess(img_bytes, data):
    image = Image.open(BytesIO(img_bytes))
    image = np.array(image)
    return image, data

import numpy as np
from PIL import Image
from io import BytesIO

def default_preprocess(img_bytes, data):
    image_path = data[0]
    image = Image.open(BytesIO(img_bytes))
    image = np.array(image.resize((256, 256)))
    return image, data
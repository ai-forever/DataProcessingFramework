from PIL import Image
from io import BytesIO

def read_image_rgb(path, force_rgb=True):
    pil_img = Image.open(path)
    pil_img.load()
    if pil_img.format is 'PNG' and pil_img.mode is not 'RGBA':
        pil_img = pil_img.convert('RGBA')
    if force_rgb:
        pil_img = pil_img.convert('RGB')
    return pil_img


def read_image_rgb_from_bytes(img_bytes, force_rgb=True):
    pil_img = Image.open(BytesIO(img_bytes))
    pil_img.load()
    if pil_img.format is 'PNG' and pil_img.mode is not 'RGBA':
        pil_img = pil_img.convert('RGBA')
    if force_rgb:
        pil_img = pil_img.convert('RGB')
    return pil_img
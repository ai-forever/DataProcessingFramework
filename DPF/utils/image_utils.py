from PIL import Image

def read_image_rgb(path, force_rgb=True):
    pil_img = Image.open(path)
    pil_img.load()
    if pil_img.format is 'PNG' and pil_img.mode is not 'RGBA':
        pil_img = pil_img.convert('RGBA')
    if force_rgb:
        pil_img = pil_img.convert('RGB')
    return pil_img
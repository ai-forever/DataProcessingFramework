from io import BytesIO

from PIL import Image


def read_image_rgb(path: str, force_rgb: bool = True) -> Image.ImageFile:
    pil_img = Image.open(path)
    pil_img.load()  # type: ignore
    if pil_img.format == "PNG" and pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    if force_rgb:
        pil_img = pil_img.convert("RGB")
    return pil_img


def read_image_rgb_from_bytes(img_bytes: bytes, force_rgb: bool = True) -> Image.ImageFile:
    pil_img = Image.open(BytesIO(img_bytes))
    pil_img.load()  # type: ignore
    if pil_img.format == "PNG" and pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    if force_rgb:
        pil_img = pil_img.convert("RGB")
    return pil_img

# test 1

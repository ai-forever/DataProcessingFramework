from typing import List, Optional
import os
import pandas as pd
import numpy as np
from scipy.fftpack import dct
import langid

from DPF.filters.utils import identical_collate_fn
from .text_filter import TextFilter


try:
    import re2 as re
except ModuleNotFoundError:
    print("Can't import package re2, using re package. It is recommended to use more efficient re2 package.")
    import re
import string

### regexs
compiled_regexs = []
compiled_regexs_2 = []

def compile_regexs_ru():
    global compiled_regexs, compiled_regexs_2
    compiled_regexs = []
    compiled_regexs_2 = []
    
    compiled_regexs.append((re.compile(r'&quot;?'), ''))
    compiled_regexs.append((re.compile(r'\d*&#\d*;\d*'), ''))
    compiled_regexs.append((re.compile(r'\.? купить за \d+ руб\.?'), ''))
    compiled_regexs.append((re.compile(r'проект \b\d+\-\d+\b'), ''))
    compiled_regexs.append((re.compile(r'проект \b\d+\w+\b'), ''))
    compiled_regexs.append((re.compile(r'\d+\s?х\s?\d*,?\.?\d+\s?\d*,?\.?\d*'), ''))
    
    # new filters
    compiled_regexs.append((re.compile(r'\b[\d\.]+\s*[xх×\-/]?\s*[\d\.]*\s*[xх×\-/]?\s*[\d\.]*\s*(?:cm|mm|m|km|inch|ct|g|kg|l|ml|w|h|px|b|kb|mb|gb|см|мм|м|км|л|грамм|кг|килограмм|в|вт|квт)\b'), '')) # 145 x 195 cm | 80km | 25x40x50 mm | 31.8/34.9mm
    compiled_regexs.append((re.compile(r'\b\w*[\-|/]?\d+[\-|/]\w*\b'), ''))
    compiled_regexs.append((re.compile(r'\b[\w]+\.ру'), ''))
    compiled_regexs.append((re.compile(r'(at )?\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))'), '')) # regex for urls
    compiled_regexs.append((re.compile(r'\.{2,}'), ' ')) # .. | ... - 2 or more dots
    compiled_regexs.append((re.compile(r'\b[а-яА-Я]{1,3}\d{3,15}\b'), '')) # jc6640
    compiled_regexs.append((re.compile(r'\(\s?#?\s?\d+\s?\)'), ' ')) # (19)
    compiled_regexs.append((re.compile(r'\b\d{1,4}[-/.]\d{1,4}[-/.]\d{1,4}\b'), '')) # 2020-11-06
    compiled_regexs.append((re.compile(r'<[\/a-zA-z\-\s]+[\w\d\/=:\\_\.\-"\s]*>'), ' ')) # </strong> - html
    #

    compiled_regexs.append((re.compile(r'артикул поставщика \d+'), ''))
    compiled_regexs.append((re.compile(r'артикул \d+'), ''))
    compiled_regexs.append((re.compile(r'@[\w\d]+\b'), ''))
    compiled_regexs.append((re.compile(r'размер \d+\-\d+'), ''))
    compiled_regexs.append((re.compile(r'рост \d+\-\d+'), ''))
    compiled_regexs.append((re.compile(r'\b[\d\.]*[xх\-]?[\d\.]*[xх\-]?[\d\.]+\s*(?:г\/кв\.м|кв\.м|мм|см|см|дм|мкм|мл|г|кг|м|л)\b'), ''))
    
    compiled_regexs.append((re.compile(r'фото\s?[-№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ фото'), ''))
    compiled_regexs.append((re.compile(r'фотография\s?[-№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ фотография'), ''))
    compiled_regexs.append((re.compile(r'изображение\s?[-№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ изображение'), ''))
    compiled_regexs.append((re.compile(r'скриншот\s?[-№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ скриншот'), ''))
    compiled_regexs.append((re.compile(r'screenshot\s?[-№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ screenshot'), ''))
    
    compiled_regexs.append((re.compile(r'фото со стока'), ''))
    compiled_regexs.append((re.compile(r'лицензионные стоковые изображения'), ''))
    compiled_regexs.append((re.compile(r'лицензионные стоковые видео'), ''))
    compiled_regexs.append((re.compile(r'лицензионные стоковые видео'), ''))
    compiled_regexs.append((re.compile(r'\b\w*[\-|/]?\d+[\-|/]\w*\b'), ''))
    compiled_regexs.append((re.compile(r'стоковый видеоролик'), ''))
    compiled_regexs.append((re.compile(r'стоковые видео и кадры b-roll'), ''))
    compiled_regexs.append((re.compile(r'стоковые фото и изображения'), ''))
    compiled_regexs.append((re.compile(r'stock video'), ''))
    compiled_regexs.append((re.compile(r'free stock photos'), ''))
    compiled_regexs.append((re.compile(r'stock illustrations'), ''))
    compiled_regexs.append((re.compile(r'стоковые видеозаписи'), ''))
    compiled_regexs.append((re.compile(r'стоковое фото'), ''))
    compiled_regexs.append((re.compile(r'cтоковое фото'), ''))
    compiled_regexs.append((re.compile(r'стоковые фото'), ''))
    compiled_regexs.append((re.compile(r'стоковые видео'), ''))
    compiled_regexs.append((re.compile(r'сток видео'), ''))
    compiled_regexs.append((re.compile(r'bекторная'), ''))
    compiled_regexs.append((re.compile(r'стоковий відеоролик'), ''))
    compiled_regexs.append((re.compile(r'стокове відео'), ''))
    compiled_regexs.append((re.compile(r'стокове фото'), ''))
    compiled_regexs.append((re.compile(r'стоковое видео'), ''))
    compiled_regexs.append((re.compile(r'стоковый вектор'), ''))
    compiled_regexs.append((re.compile(r'стоковое изображение'), ''))
    compiled_regexs.append((re.compile(r'стоковая картинка'), ''))
    compiled_regexs.append((re.compile(r'стоковая'), ''))
    compiled_regexs.append((re.compile(r'иллюстрации'), ''))
    compiled_regexs.append((re.compile(r'фото шаг \d+'), ''))
    compiled_regexs.append((re.compile(r'шаг\s?[№#]?\s?\d+'), ''))
    compiled_regexs.append((re.compile(r'интернет[-\s]+магазин[\w\W]*'), ''))
    compiled_regexs.append((re.compile(r'(купите в )?интернет[-\s]+магазине[\w\W]*'), ''))
    compiled_regexs.append((re.compile(r'ярмарка мастеров'), ''))
    compiled_regexs.append((re.compile(r'youtube'), ''))
    # compiled_regexs.append((re.compile(r'карточка пользователя [\w\W]+'), ''))
    compiled_regexs.append((re.compile(r'вконтакте'), ''))
    compiled_regexs.append((re.compile(r'(риа новости).*$'), ''))
    compiled_regexs.append((re.compile(r'авито'), ''))
    compiled_regexs.append((re.compile(r'avito'), ''))
    compiled_regexs.append((re.compile(r'анкета знакомств[\w\W]*'), ''))
    compiled_regexs.append((re.compile(r'яндекс[\.\s]новости[\w\W]*'), ''))
    compiled_regexs.append((re.compile(r'яндекс[\.\s]дзен[\w\W]*'), ''))
    compiled_regexs.append((re.compile(r'яндекс\.\w+'), ''))
    compiled_regexs.append((re.compile(r'профиль в вк'), ' '))
    compiled_regexs.append((re.compile(r'заказать на ярмарке мастеров'), ' '))
    compiled_regexs.append((re.compile(r'бесплатно'), ' '))
    compiled_regexs.append((re.compile(r'скачать обои'), ' '))
    compiled_regexs.append((re.compile(r'скачать'), ' '))
    compiled_regexs.append((re.compile(r'фото и отзывы'), ' '))
    compiled_regexs.append((re.compile(r'описание, цена, фото'), ' '))
    compiled_regexs.append((re.compile(r'отзывы, характеристики, фото'), ' '))
    compiled_regexs.append((re.compile(r'предложение:'), ' '))
    compiled_regexs.append((re.compile(r'куплю:'), ' '))
    compiled_regexs.append((re.compile(r'приму в дар:'), ' '))
    compiled_regexs.append((re.compile(r'отдам даром:'), ' '))
    compiled_regexs.append((re.compile(r'отдам даром'), ' '))
    compiled_regexs.append((re.compile(r'создать мем:'), ' '))
    compiled_regexs.append((re.compile(r'[a-zA-Z]+ арт'), ' '))
    compiled_regexs.append((re.compile(r'страница\s\d+'), ' '))
    compiled_regexs.append((re.compile(r'рисовач\s.ру'), ' '))
    compiled_regexs.append((re.compile(r'объявления в [а-яА-Яa-zA-Z-]+'), ' '))
    compiled_regexs.append((re.compile(r'объявления на [а-яА-Яa-zA-Z-]+'), ' '))
    compiled_regexs.append((re.compile(r'купить со скидкой'), ''))
    compiled_regexs.append((re.compile(r'купить, цена в москве'), ''))
    compiled_regexs.append((re.compile(r'социальная сеть фотокто'), ''))
    compiled_regexs.append((re.compile(r'\- красивые картинки'), ''))
    # compiled_regexs.append((re.compile(r'\- купить в [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'самые лучшие, фото [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'рецепт с фото пошагово [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'\: фото и описание [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'обсуждение на liveinternet [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'купить по лучшей цене [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'купить в [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'забронировать отель [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'перейти на официальный сайт [\w\W]+'), ''))
    # compiled_regexs.append((re.compile(r'официальный сайт [\w\W]+'), ''))
    compiled_regexs.append((re.compile(r'\b[\w]+\.ру'), ''))
    compiled_regexs.append((re.compile(r'вид \d+'), ''))
    compiled_regexs.append((re.compile(r'\b[\d\_\.\-]+[a-z]+[\d\_\.\-]+\b'), ''))
    compiled_regexs.append((re.compile(r'\b[a-z\_\.\-]+\-?[\d\_\.\-]+[a-z\_\.\-]*\b'), ''))
    compiled_regexs.append((re.compile(r'\d{5,}'), ''))
    compiled_regexs.append((re.compile(r'\/'), ', '))
    compiled_regexs.append((re.compile(r'image \d+'), ''))
    compiled_regexs.append((re.compile(r'rf$'), ''))

    compiled_regexs.append((re.compile(r'\.*\s*(?:\|?/?фото\:|//) [\w\W]+\.(?:ru|com|net|tv|\w{2,3})\s*\.*'), ''))

    compiled_regexs.append((re.compile(r'https?\S+'), ''))
    compiled_regexs.append((re.compile(r'@[\S]+\b'), ''))
    compiled_regexs.append((re.compile(r'(\s*\b[\-a-z]+\b\s*){2,}'), ' '))
    compiled_regexs.append((re.compile(r'\/\d*,\d+\w*\b'), ' '))
    compiled_regexs.append((re.compile(r'\- смотреть фильм онлайн без регистрации'), ''))
    compiled_regexs.append((re.compile(r'купить'), ''))

    compiled_regexs.append((re.compile(r'[\(\)]'), ''))
    compiled_regexs.append((re.compile(r'\s+'), ' '))
    compiled_regexs.append((re.compile(r'[\"\']{2,}'), r''))
    
    ####
            
    compiled_regexs_2.append((re.compile(r'[\(\)]'), ''))
    compiled_regexs_2.append((re.compile(r'\s+'), ' '))
    
    
def compile_regexs_eng():
    global compiled_regexs, compiled_regexs_2
    compiled_regexs = []
    compiled_regexs_2 = []
    
    compiled_regexs.append((re.compile(r'&quot;?'), ''))
    compiled_regexs.append((re.compile(r'\d*&#\d*;\d*'), ''))

    compiled_regexs.append((re.compile(r'@[\w\d]+\b'), ''))
    compiled_regexs.append((re.compile(r'\b[\d\.]+\s*[xх×\-/]?\s*[\d\.]*\s*[xх×\-/]?\s*[\d\.]*\s*(?:cm|mm|m|km|inch|ct|g|kg|l|ml|w|h|px)\b'), '')) # 145 x 195 cm | 80km | 25x40x50 mm | 31.8/34.9mm
    compiled_regexs.append((re.compile(r'\b\w*[\-|/]?\d+[\-|/]\w*\b'), ''))
    compiled_regexs.append((re.compile(r'\b[\w]+\.ру'), ''))
    compiled_regexs.append((re.compile(r'(at )?\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))'), '')) # regex for urls
    compiled_regexs.append((re.compile(r'\.{2,}'), ' ')) # .. | ... - 2 or more dots
    compiled_regexs.append((re.compile(r'\b[a-zA-Z]{1,3}\d{3,15}\b'), '')) # jc6640
    compiled_regexs.append((re.compile(r'\(?(serial\s?)?#?\d{5,}\)?'), ''))
    compiled_regexs.append((re.compile(r'\(\s?#?\s?\d+\s?\)'), ' ')) # (19)
    compiled_regexs.append((re.compile(r'\b\d{1,4}[-/.]\d{1,4}[-/.]\d{1,4}\b'), '')) # 2020-11-06
    compiled_regexs.append((re.compile(r'<[\/a-zA-z\-\s]+[\w\d\/=:\\_\.\-"\s]*>'), ' ')) # </strong> - html
    compiled_regexs.append((re.compile(r'\/'), ', '))
    compiled_regexs.append((re.compile(r'::'), ' - '))
    
    compiled_regexs.append((re.compile(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b'), '')) # 300x250, 18x9.5
    compiled_regexs.append((re.compile(r'\b[\w\d]+\.(png|jpg|jpeg|bmp|webp|pdf|apk|eps|mp4)\b'), '')) # image names
    compiled_regexs.append((re.compile(r'(for\s)?[$€]\d+[\.,]?\s?\d*(\s?usd)?'), '')) # for $350.00 | $8.79 USD
    compiled_regexs.append((re.compile(r'\bx\b[^-_\.,!?]'), ' '))
    
    compiled_regexs.append((re.compile(r'(royalty\s)?(free\s)?(stock\s(photo|image)[\w]*)\b'), ''))
    compiled_regexs.append((re.compile(r'royalty free image'), ''))
    compiled_regexs.append((re.compile(r'stock footage video'), ''))
    compiled_regexs.append((re.compile(r'stock footage'), ''))
    compiled_regexs.append((re.compile(r'stockfoto'), ''))
    compiled_regexs.append((re.compile(r'photostock'), ''))
    compiled_regexs.append((re.compile(r'foto[\s]?stock'), ''))
    compiled_regexs.append((re.compile(r'foto de stock'), ''))
    compiled_regexs.append((re.compile(r'depositphoto[s]?'), ''))
    compiled_regexs.append((re.compile(r'istock'), ''))
    compiled_regexs.append((re.compile(r'shutterstock'), ''))
    compiled_regexs.append((re.compile(r'stok fotoğraf'), ''))
    compiled_regexs.append((re.compile(r'printsalon'), ''))
    compiled_regexs.append((re.compile(r'(on\s)?pinterest'), ''))
    compiled_regexs.append((re.compile(r'photographie de stock - premium libres de droits, code'), ''))
    compiled_regexs.append((re.compile(r'photographie de stock'), ''))
    compiled_regexs.append((re.compile(r'(by\s)?stocktrek images'), ''))
    compiled_regexs.append((re.compile(r'\bebook\b'), ''))
    compiled_regexs.append((re.compile(r'lang_evoimages'), ''))
    compiled_regexs.append((re.compile(r'image result for'), ''))
    compiled_regexs.append((re.compile(r'discover now at'), ''))
    compiled_regexs.append((re.compile(r'\bmls\b'), ''))
    compiled_regexs.append((re.compile(r'getty images'), ''))
    compiled_regexs.append((re.compile(r'image credit'), ''))
    compiled_regexs.append((re.compile(r'zipsite'), ''))
    
    compiled_regexs.append((re.compile(r'click to see full-size photo viewer'), ''))
    compiled_regexs.append((re.compile(r'click image for larger version'), ''))
    compiled_regexs.append((re.compile(r'click here to view larger image'), ''))
    compiled_regexs.append((re.compile(r'click here to see product details'), ''))
    compiled_regexs.append((re.compile(r'click on [\w\s]+ to close'), ''))
    compiled_regexs.append((re.compile(r'click here for more'), ''))
    compiled_regexs.append((re.compile(r'click ((photo)|(image)|(here)) ((for)|(to)) \w+'), ''))
    compiled_regexs.append((re.compile(r'click ((for)|(to)) \w+'), ''))
    
    compiled_regexs.append((re.compile(r'(online\s)?(shop[\w]*\s)?(buy)\s?(low)?'), ''))
    compiled_regexs.append((re.compile(r"product('s)? image \d+"), ''))
    compiled_regexs.append((re.compile(r'google search'), ''))
    compiled_regexs.append((re.compile(r'\bpreview\b'), ''))
    compiled_regexs.append((re.compile(r'permalink to'), ''))
    compiled_regexs.append((re.compile(r'\bfile\b'), ''))
    compiled_regexs.append((re.compile(r'stock video'), ''))
    compiled_regexs.append((re.compile(r'free stock photos'), ''))
    compiled_regexs.append((re.compile(r'stock illustrations'), ''))
    compiled_regexs.append((re.compile(r'youtube'), ''))
    compiled_regexs.append((re.compile(r'reddit'), ''))
    compiled_regexs.append((re.compile(r'pictures & photos'), ''))
    compiled_regexs.append((re.compile(r'image\s-?\s?\d+(\sof\s\d+)?'), ''))
    compiled_regexs.append((re.compile(r'photo\s-?\s?\d+(\sof\s\d+)?'), ''))
    compiled_regexs.append((re.compile(r'picture\s-?\s?\d+(\sof\s\d+)?'), ''))
    compiled_regexs.append((re.compile(r'gallery\s-?\s?\d+(\sof\s\d+)?'), ''))
    compiled_regexs.append((re.compile(r'view\s-?\s?\d+(\sof\s\d+)?'), ''))
    compiled_regexs.append((re.compile(r'\d+ photo(s)?'), ''))
    compiled_regexs.append((re.compile(r'\d+ image(s)?'), ''))
    compiled_regexs.append((re.compile(r'pack \d+'), ''))
    compiled_regexs.append((re.compile(r'screenshot(s)? \d+'), ''))
    compiled_regexs.append((re.compile(r'\d+ screenshot(s)?'), ''))
    compiled_regexs.append((re.compile(r'hd video'), ''))
    compiled_regexs.append((re.compile(r'\bvideo\b'), ''))
    compiled_regexs.append((re.compile(r'price for sale'), ''))
    compiled_regexs.append((re.compile(r'for sale( of)?'), ''))
    compiled_regexs.append((re.compile(r'wholesale'), ''))
    compiled_regexs.append((re.compile(r'(worldwide\s)?(free\s)?shipping'), ''))
    compiled_regexs.append((re.compile(r'(free\s)?download(\sfree)?'), ''))
    compiled_regexs.append((re.compile(r'\bclick\b\s(?:for|on)\s\w+'), ''))
    compiled_regexs.append((re.compile(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?'), ''))
    compiled_regexs.append((re.compile(r'stock[_\d]+'), ''))
    compiled_regexs.append((re.compile(r'\bpage \d+\b'), ''))
    compiled_regexs.append((re.compile(r'hi-res'), ''))

    compiled_regexs.append((re.compile(r'https?\S+'), ''))
    compiled_regexs.append((re.compile(r'@[\S]+\b'), ''))
    compiled_regexs.append((re.compile(r'\/\d*,\d+\w*\b'), ' '))

    compiled_regexs.append((re.compile(r'[\(\)]'), ' '))
    compiled_regexs.append((re.compile(r'\s+'), ' '))
    compiled_regexs.append((re.compile(r'[\"\']{2,}'), r''))
    
    compiled_regexs_2.append((re.compile(r'[\(\)]'), ''))
    compiled_regexs_2.append((re.compile(r'\s+'), ' '))
    compiled_regexs_2.append((re.compile(r'\bby\s*$'), ''))
    

### additional regexs
html_pattern = re.compile(r"<.*?>|&([a-zA-Z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});")
def remove_html(text):
    """
        Remove the html in sample text
    """
    return html_pattern.sub(" ", text)

emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
def remove_special_characters(text):
    """
        Remove special special characters, including symbols, emojis, and other graphic characters
    """
    return emoji_pattern.sub(r' ', text)

decontracted_patterns = [
    (re.compile(r"won\'t"), r"will not"),
    (re.compile(r"can\'t"), r"can not"),
    (re.compile(r"\'ll"), r" would"),
    (re.compile(r"\'ve"), r" have"),
]
def decontracted(phrase):
    lower_phrase = str(phrase).lower().strip()
    for pattern, replacement in decontracted_patterns:
        iterator = reversed(list(pattern.finditer(lower_phrase)))
        for match in iterator:
            pos = list(match.span())
            phrase = phrase[:pos[0]] + replacement + phrase[pos[1]:]
        lower_phrase = str(phrase).lower().strip()
        phrase = pattern.sub(replacement, phrase)
    return phrase

additional_patterns = [
    (re.compile(r"\n"), r" "),
    (re.compile(r"\#\d+"), r" "),
    (re.compile(r"\b\d{3,}[a-zA-z]+\d*\b"), r" "),
    (re.compile(r"\b[a-zA-z]{3,}\d+\w*\b"), r" "),
    (re.compile(r"\[\w+\]"), r" "),
    (re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"), r" "),
    (re.compile(r"(http://.*?\s)|(http://.*)"), r" "),
    (re.compile(r"\s+"), r" "),
    (re.compile(r"(\-\s\-)+"), r"-"),
]
def additional_cleaning(text):
    text = remove_html(text)
    text = remove_special_characters(text)
    text = decontracted(text)
    
    for pattern, replacement in additional_patterns:
        text = pattern.sub(replacement, text)
    return text.strip()


### main code
def clean_caption(caption):
    lower_caption = str(caption).lower().strip()
    for re_compiled, replacement in compiled_regexs:
        iterator = reversed(list(re_compiled.finditer(lower_caption)))
        for match in iterator:
            pos = list(match.span())
            caption = caption[:pos[0]] + replacement + caption[pos[1]:]
        lower_caption = str(caption).lower().strip()

    caption = caption.strip()
    lower_caption = str(caption).lower().strip()
    
    if caption and caption[0] in string.punctuation + '—)':
        caption = caption[1:].strip()
        if caption and caption[0] in string.punctuation + '—)':
            caption = caption[1:].strip()
    if caption and caption[-1] in string.punctuation + '—(':
        caption = caption[:-1].strip()
        if caption and caption[-1] in string.punctuation + '—(':
            caption = caption[:-1].strip()
            
    lower_caption = str(caption).lower().strip()
    for re_compiled, replacement in compiled_regexs_2:
        iterator = reversed(list(re_compiled.finditer(lower_caption)))
        for match in iterator:
            pos = list(match.span())
            caption = caption[:pos[0]] + replacement + caption[pos[1]:]
        lower_caption = str(caption).lower().strip()

        caption = caption.strip()
    
    caption = additional_cleaning(caption)

    return caption


def clean_joined_words(s):
    if len(s.split(' ')) == 1:
        replaced_s = s.replace('-', ' ')
        if len(replaced_s) != 1:
            return replaced_s
    return s


class RegexFilter(TextFilter):
    def __init__(
         self, 
         caption_name:str='caption',
         is_regex_ru: bool = True):
            
        super(RegexFilter, self).__init__(caption_name)
        self.result_columns = 'clean_caption'
        
        if is_regex_ru:
            compile_regexs_ru()
        else:
            compile_regexs_eng()
     
          
    def filter_text(self, row):
        caption = clean_caption(row[self.caption_name])
        return caption
        
# TODO(review) - значения похожи на константы, константы лучше именовать капсом (RU_REGEXS, ENG_REGEXS и т.д)
ru_regexs = [
    (r"&quot;?", ""),
    (r"\d*&#\d*;\d*", ""),
    (r"\.? купить за \d+ руб\.?", ""),
    (r"проект \b\d+\-\d+\b", ""),
    (r"проект \b\d+\w+\b", ""),
    (r"\d+\s?х\s?\d*,?\.?\d+\s?\d*,?\.?\d*", ""),
    (
        r"\b[\d\.]+\s*[xх×\-/]?\s*[\d\.]*\s*[xх×\-/]?\s*[\d\.]*\s*(?:cm|mm|m|km|inch|ct|g|kg|l|ml|w|h|px|b|kb|mb|gb|см|мм|м|км|л|грамм|кг|килограмм|в|вт|квт)\b",
        "",
    ),
    (r"\b\w*[\-|/]?\d+[\-|/]\w*\b", ""),
    (r"\b[\w]+\.ру", ""),
    (
        r"(at )?\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
        "",
    ),
    (r"\.{2,}", " "),
    (r"\b[а-яА-Я]{1,3}\d{3,15}\b", ""),
    (r"\(\s?#?\s?\d+\s?\)", " "),
    (r"\b\d{1,4}[-/.]\d{1,4}[-/.]\d{1,4}\b", ""),
    (r'<[\/a-zA-z\-\s]+[\w\d\/=:\\_\.\-"\s]*>', " "),
    (r"артикул поставщика \d+", ""),
    (r"артикул \d+", ""),
    (r"@[\w\d]+\b", ""),
    (r"размер \d+\-\d+", ""),
    (r"рост \d+\-\d+", ""),
    (
        r"\b[\d\.]*[xх\-]?[\d\.]*[xх\-]?[\d\.]+\s*(?:г\/кв\.м|кв\.м|мм|см|см|дм|мкм|мл|г|кг|м|л)\b",
        "",
    ),
    (r"фото\s?[-№#]?\s?\d+", ""),
    (r"\d+ фото", ""),
    (r"фотография\s?[-№#]?\s?\d+", ""),
    (r"\d+ фотография", ""),
    (r"изображение\s?[-№#]?\s?\d+", ""),
    (r"\d+ изображение", ""),
    (r"скриншот\s?[-№#]?\s?\d+", ""),
    (r"\d+ скриншот", ""),
    (r"screenshot\s?[-№#]?\s?\d+", ""),
    (r"\d+ screenshot", ""),
    (r"фото со стока", ""),
    (r"лицензионные стоковые изображения", ""),
    (r"лицензионные стоковые видео", ""),
    (r"лицензионные стоковые видео", ""),
    (r"\b\w*[\-|/]?\d+[\-|/]\w*\b", ""),
    (r"стоковый видеоролик", ""),
    (r"стоковые видео и кадры b-roll", ""),
    (r"стоковые фото и изображения", ""),
    (r"stock video", ""),
    (r"free stock photos", ""),
    (r"stock illustrations", ""),
    (r"стоковые видеозаписи", ""),
    (r"стоковое фото", ""),
    (r"cтоковое фото", ""),
    (r"стоковые фото", ""),
    (r"стоковые видео", ""),
    (r"сток видео", ""),
    (r"bекторная", ""),
    (r"стоковий відеоролик", ""),
    (r"стокове відео", ""),
    (r"стокове фото", ""),
    (r"стоковое видео", ""),
    (r"стоковый вектор", ""),
    (r"стоковое изображение", ""),
    (r"стоковая картинка", ""),
    (r"стоковая", ""),
    (r"иллюстрации", ""),
    (r"фото шаг \d+", ""),
    (r"шаг\s?[№#]?\s?\d+", ""),
    (r"интернет[-\s]+магазин[\w]*", ""),
    (r"(купите в )?интернет[-\s]+магазине[\w]*", ""),
    (r"ярмарка мастеров", ""),
    (r"youtube", ""),
    (r"вконтакте", ""),
    (r"(риа новости).*$", ""),
    (r"авито", ""),
    (r"avito", ""),
    (r"анкета знакомств[\w]*", ""),
    (r"яндекс[\.\s]новости[\w]*", ""),
    (r"яндекс[\.\s]дзен[\w]*", ""),
    (r"яндекс\.\w+", ""),
    (r"профиль в вк", " "),
    (r"заказать на ярмарке мастеров", " "),
    (r"бесплатно", " "),
    (r"скачать обои", " "),
    (r"скачать", " "),
    (r"фото и отзывы", " "),
    (r"описание, цена, фото", " "),
    (r"отзывы, характеристики, фото", " "),
    (r"предложение:", " "),
    (r"куплю:", " "),
    (r"приму в дар:", " "),
    (r"отдам даром:", " "),
    (r"отдам даром", " "),
    (r"создать мем:", " "),
    (r"[a-zA-Z]+ арт", " "),
    (r"страница\s\d+", " "),
    (r"рисовач\s.ру", " "),
    (r"объявления в [а-яА-Яa-zA-Z-]+", " "),
    (r"объявления на [а-яА-Яa-zA-Z-]+", " "),
    (r"купить со скидкой", ""),
    (r"купить, цена в москве", ""),
    (r"социальная сеть фотокто", ""),
    (r"\- красивые картинки", ""),
    (r"телефон", ""),
    (r"[-№#]?\s?заказать", ""),
]

eng_regexs = [
    (r"\b\w*[\-|/]?\d+[\-|/]\w*\b", ""),
    (r"\b[\w]+\.ру", ""),
    (
        r"(at )?\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
        "",
    ),
    (r"\.{2,}", " "),
    (r"\b[а-яА-Я]{1,3}\d{3,15}\b", ""),
    (r"\(\s?#?\s?\d+\s?\)", " "),
    (r"\b\d{1,4}[-/.]\d{1,4}[-/.]\d{1,4}\b", ""),
    (r'<[\/a-zA-z\-\s]+[\w\d\/=:\\_\.\-"\s]*>', " "),
    (r"\b[\w]+\.ру", ""),
    (r"вид \d+", ""),
    (r"\b[\d\_\.\-]+[a-z]+[\d\_\.\-]+\b", ""),
    (r"\b[a-z\_\.\-]+\-?[\d\_\.\-]+[a-z\_\.\-]*\b", ""),
    (r"\d{5,}", ""),
    (r"\/", ", "),
    (r"image \d+", ""),
    (r"rf$", ""),
    (r"\.*\s*(?:\|?/?фото\:|//) [\w\W]+\.(?:ru|com|net|tv|\w{2,3})\s*\.*", ""),
    (r"https?\S+", ""),
    (r"@[\S]+\b", ""),
    (r"(\s*\b[\-a-z]+\b\s*){2,}", " "),
    (r"\/\d*,\d+\w*\b", " "),
    (r"\- смотреть фильм онлайн без регистрации", ""),
    (r"купить", ""),
    (r"[\(\)]", ""),
    (r"\s+", " "),
    (r"[\"']{2,}", ""),
    (r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", ""),
    (r"\b[\w\d]+\.(png|jpg|jpeg|bmp|webp|pdf|apk|eps|mp4)\b", ""),
    (r"(for\s)?[$€]\d+[\.,]?\s?\d*(\s?usd)?", ""),
    (r"\bx\b[^-_\.,!?]", " "),
    (r"(royalty\s)?(free\s)?(stock\s(photo|image)[\w]*)\b", ""),
    (r"royalty free image", ""),
    (r"stock footage video", ""),
    (r"stock footage", ""),
    (r"stockfoto", ""),
    (r"photostock", ""),
    (r"foto[\s]?stock", ""),
    (r"foto de stock", ""),
    (r"depositphoto[s]?", ""),
    (r"istock", ""),
    (r"shutterstock", ""),
    (r"stok fotoğraf", ""),
    (r"printsalon", ""),
    (r"(on\s)?pinterest", ""),
    (r"photographie de stock - premium libres de droits, code", ""),
    (r"photographie de stock", ""),
    (r"(by\s)?stocktrek images", ""),
    (r"\bebook\b", ""),
    (r"lang_evoimages", ""),
    (r"image result for", ""),
    (r"discover now at", ""),
    (r"\bmls\b", ""),
    (r"getty images", ""),
    (r"image credit", ""),
    (r"zipsite", ""),
    (r"click to see full-size photo viewer", ""),
    (r"click image for larger version", ""),
    (r"click here to view larger image", ""),
    (r"click here to see product details", ""),
    (r"click on [\w\s]+ to close", ""),
    (r"click here for more", ""),
    (r"click ((photo)|(image)|(here)) ((for)|(to)) \w+", ""),
    (r"click ((for)|(to)) \w+", ""),
    (r"(online\s)?(shop[\w]*\s)?(buy)\s?(low)?", ""),
    (r"product('s)? image \d+", ""),
    (r"google search", ""),
    (r"\bpreview\b", ""),
    (r"permalink to", ""),
    (r"\bfile\b", ""),
    (r"stock video", ""),
    (r"free stock photos", ""),
    (r"stock illustrations", ""),
    (r"youtube", ""),
    (r"reddit", ""),
    (r"pictures & photos", ""),
    (r"image\s-?\s?\d+(\sof\s\d+)?", ""),
    (r"photo\s-?\s?\d+(\sof\s\d+)?", ""),
    (r"picture\s-?\s?\d+(\sof\s\d+)?", ""),
    (r"gallery\s-?\s?\d+(\sof\s\d+)?", ""),
    (r"view\s-?\s?\d+(\sof\s\d+)?", ""),
    (r"\d+ photo(s)?", ""),
    (r"\d+ image(s)?", ""),
    (r"pack \d+", ""),
    (r"screenshot(s)? \d+", ""),
    (r"\d+ screenshot(s)?", ""),
    (r"hd video", ""),
    (r"\bvideo\b", ""),
    (r"price for sale", ""),
    (r"for sale( of)?", ""),
    (r"wholesale", ""),
    (r"(worldwide\s)?(free\s)?shipping", ""),
    (r"(free\s)?download(\sfree)?", ""),
    (r"\bclick\b\s(?:for|on)\s\w+", ""),
    (r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", ""),
    (r"stock[_\d]+", ""),
    (r"\bpage \d+\b", ""),
    (r"hi-res", ""),
    (r"https?\S+", ""),
    (r"@[\S]+\b", ""),
    (r"\/\d*,\d+\w*\b", " "),
    (r"[\(\)]", " "),
    (r"\s+", " "),
    (r"[\"']{2,}", ""),
]

emoji_regexs = [
    (
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        " ",
    )
]

special_regexs = [
    (r"\n", " "),
    (r"\#\d+", " "),
    (r"\b\d{3,}[a-zA-z]+\d*\b", " "),
    (r"\b[a-zA-z]{3,}\d+\w*\b", " "),
    (r"\[\w+\]", " "),
    (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " "),
    (r"https?\S+", ""),
    (r"(http://.*?\s)|(http://.*)", " "),
    (r"\s+", " "),
    (r"(\-\s\-)+", "-"),
    ("won't", "will not"),
    ("can't", "can not"),
    ("'ll", " would"),
    ("'ve", r" have"),
]

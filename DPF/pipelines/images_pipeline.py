import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from DPF.dataloaders.images import UniversalT2IDataloader
from DPF.filters.images.img_filter import ImageFilter
from DPF.filters.text2image.t2ifilter import T2IFilter
from DPF.filesystems.filesystem import FileSystem

    
class FilterPipeline:
    def __init__(self):
        pass
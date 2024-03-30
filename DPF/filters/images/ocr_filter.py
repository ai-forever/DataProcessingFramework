import json
import os
from typing import Any, Optional

import torch

from DPF.utils import read_image_rgb_from_bytes

from ...types import ModalityToDataMapping
from .img_filter import ImageFilter
from .ocr_model.dataset import AlignCollate
from .ocr_model.model import Model
from .ocr_model.utils import AttnLabelConverter


class Options:
    pass


class OCRFilter(ImageFilter):

    def __init__(
        self,
        weights_path: str,
        model_name: Optional[str] = None,
        device: str = "cuda:0",
        workers: int = 16,
        pad: int = 5,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        self.num_workers = workers
        self.batch_size = 1
        self.device = device

        self.weights_path = weights_path
        self.model_name = model_name or os.path.basename(self.weights_path).split('.')[0]
        # load model
        self.opt = Options()
        self.opt.workers = 4
        self.opt.batch_size = 192
        self.opt.batch_max_length = 32
        self.opt.imgH = 32
        self.opt.imgW = 100
        self.opt.rgb = False
        self.opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.opt.sensitive = False
        self.opt.PAD = False
        self.opt.Transformation = "TPS"
        self.opt.FeatureExtraction = "ResNet"
        self.opt.SequenceModeling = "BiLSTM"
        self.opt.Prediction = "Attn"
        self.opt.num_fiducial = 20
        self.opt.input_channel = 1
        self.opt.output_channel = 512
        self.opt.hidden_size = 256

        self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        self.model = Model(self.opt)
        weights = torch.load(self.weights_path)
        keys = list(weights.keys())
        for key in keys:
            weights[key.lstrip('module.')] = weights[key]
            weights.pop(key)

        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

        self.AlignCollate = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        #
        self.text_box_col = "text_boxes"
        self.ocr_col = f"OCR_{self.model_name}"

    @property
    def result_columns(self) -> list[str]:
        return [self.ocr_col]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "preprocess_f": self.preprocess_data,
            "drop_last": False,
            "cols_to_return": [self.text_box_col],
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        image_path = metadata["image_path"]
        boxes = json.loads(metadata[self.text_box_col])
        pil_img = read_image_rgb_from_bytes(modality2data['image']).convert('L')
        return image_path, pil_img, boxes

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()
        image_path, pil_img, boxes = batch[0]
        w, h = pil_img.size

        input_data = []
        for box in boxes:
            left = max(box[0][0], 0)
            upper = max(box[0][1], 0)
            right = min(box[1][0], w)
            lower = min(box[1][1], h)
            if upper > lower:
                upper, lower = lower, upper
            if left > right:
                left, right = right, left

            crop = pil_img.crop(
                (left, upper, right, lower)
            )
            input_data.append((crop, ''))

        if len(input_data) == 0:
            df_batch_labels[self.ocr_col].append("[]")
            df_batch_labels["image_path"].append(image_path)
            return df_batch_labels

        data_preproc = self.AlignCollate(input_data)
        image_tensors = data_preproc[0]

        batch_size = image_tensors.size(0)
        image = image_tensors.to(self.device)
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

        preds = self.model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        preds_str = [s.replace('[s]', '') for s in preds_str]

        res = []
        for box, prediction in zip(boxes, preds_str):
            res.append((box, prediction))

        df_batch_labels[self.ocr_col].append(json.dumps(res))
        df_batch_labels["image_path"].append(image_path)

        return df_batch_labels

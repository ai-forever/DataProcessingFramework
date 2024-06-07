import os
from io import BytesIO
from typing import Any, Optional

import torch
import wget

from DPF.types import ModalityToDataMapping

from .grounding_gpt.lego import conversation as conversation_lib
from .grounding_gpt.lego.constants import (
    DEFAULT_VIDEO_END_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VIDEO_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .grounding_gpt.lego.conversation import SeparatorStyle
from .grounding_gpt.lego.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from .grounding_gpt.lego.model.builder import CONFIG, load_pretrained_model
from .grounding_gpt.video_llama.processors.video_processor import load_video
from .video_filter import VideoFilter

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
BLIP_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"


class GroundingGPTFilter(VideoFilter):
    """
    GroundingGPT inference class to get captions for auto-labeling videos.
    More info about the model here: https://github.com/lzw-lzw/GroundingGPT
    """
    def __init__(
        self,
        weights_path: str = "zwli/GroundingGPT",
        additional_files_folder: str = './ckpt',
        prompt: str = "detailed_video",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        device: str = "cuda:0",
        workers: int = 16,
        batch_size: int = 8,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.model_name = get_model_name_from_path(weights_path)
        self.prompt_to_use = prompt
        prompt_templates = {
            'detailed_video': 'Describe this video and its style in a very detailed manner',
            'grounding_gpt': 'Offer a detailed explanation of of the clip’s key features',
            'short_video': 'Describe this video and its style briefly'
        }

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.inp = prompt_templates[self.prompt_to_use]
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if not os.path.exists(additional_files_folder):
            os.makedirs(additional_files_folder)
            os.makedirs(os.path.join(additional_files_folder, 'imagebind'))
            wget.download(IMAGEBIND_URL, out=os.path.join(additional_files_folder, 'imagebind'))
            wget.download(BLIP_URL, out=additional_files_folder)

        pretrainers = load_pretrained_model(weights_path, device=self.device)  # type: ignore
        self.model, self.tokenizer, self.image_processor, self.video_transform, self.context_len = pretrainers

        self.conv = conversation_lib.default_conversation.copy()

        inp = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len + DEFAULT_VIDEO_END_TOKEN + '\n' + self.inp
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        self.input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        self.stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [self.stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    @property
    def result_columns(self) -> list[str]:
        return [f"caption {self.model_name} prompt {self.prompt_to_use}"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        video_file = BytesIO(modality2data['video'])

        video_file = load_video(video_path=video_file, n_frms=self.model.config.max_frame,
                                height=224, width=224, sampling="uniform", return_msg=False)
        video_tensor = self.video_transform(video_file).unsqueeze(0)
        return key, video_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, video_tensors = list(zip(*batch))
        video_tensors_batch = default_collate(video_tensors).to(self.device, torch.bfloat16)  # type: ignore
        input_ids_batch = self.input_ids.repeat_interleave(video_tensors_batch.shape[0], 0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch,
                images=None,
                videos=video_tensors_batch,
                sounds=None,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=0.85,
                no_repeat_ngram_size=2,
                use_cache=True,
                device=self.device
            )

        all_outputs: list[Optional[str]] = []
        for i in range(output_ids.shape[0]):
            caption = self.tokenizer.decode(output_ids[i, self.input_ids.shape[1]:]).strip()
            if caption.endswith(self.stop_str):
                caption = caption[:-len(self.stop_str)]
            all_outputs.append(caption)
        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)
        return df_batch_labels

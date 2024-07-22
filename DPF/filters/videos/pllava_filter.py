import os
from io import BytesIO
from typing import Any, Optional

import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from PIL import Image

from DPF.filters.videos.video_filter import VideoFilter
from DPF.types import ModalityToDataMapping

from .pllava_filter_core.tasks.eval.eval_utils import conv_templates
from .pllava_filter_core.tasks.eval.model_utils import load_pllava


def get_index(num_frames: int, num_segments: int) -> np.ndarray[Any, Any]:
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])


def load_video(video_bytes: BytesIO, num_segments: int = 8, return_msg: bool = False, num_frames: int = 16, resolution: int = 336) -> Any:
    transforms = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_bytes, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    images_group = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())  # type: ignore
        images_group.append(transforms(img))
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return images_group, msg
    else:
        return images_group


class PllavaFilter(VideoFilter):
    """
    Pllava inference class to get captions for videos.
    More info about the model here: https://pllava.github.io
    """
    def __init__(
        self,
        model_path: str,
        weights_path: str,
        weights_dir: str,
        prompt: str = "short",
        prompts: Optional[dict[str, str]] = None,
        do_sample: bool = True,
        batch_size: int = 16,
        conv_mode: str = 'eval_vcg_llavanext',
        device: str = "cuda:0",
        workers: int = 16,
        num_frames: int = 32,
        max_new_tokens: int = 100,
        num_segments: int = 32,
        resolution: int = 672,
        temperature: float = 0.1,
        use_lora: bool = True,
        lora_alpha: int = 4,
        pbar: bool = True,
        _pbar_position: int = 0,
        use_multi_gpus: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(pbar, _pbar_position)
        self.weights_dir = weights_dir
        self.max_new_tokens = max_new_tokens
        self.conv_mode = conv_mode
        self.use_lora = use_lora
        self.do_sample = do_sample
        self.lora_alpha = lora_alpha
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.num_segments = batch_size
        self.num_workers = workers
        self.device = device
        self.prompt_to_use = prompt
        self.temperature = temperature
        self.resolution = resolution
        self.num_segments = num_segments
        self.num_frames = num_frames
        self.use_cache = use_cache
        self.use_multi_gpus = use_multi_gpus

        self.model_name = model_path.split('/')[-1]

        if prompts is None:
            self.prompts = {
                'detailed_video': 'Please provide a caption for this image. Speak confidently and describe everything clearly. Do not lie and                        describe only what you can see',
                'pixart': 'Describe this image and its style in a very detailed manner',
                'short': 'Describe this image very shortly in 1-2 short sentences',
                'short-video': 'Describe this video very shortly in 1-2 short sentences. Describe what is happening in this video.'
            }
        else:
            self.prompts = prompts

        self.input_ids = self.prompts[self.prompt_to_use]

        self.conv = conv_templates[self.conv_mode].copy()  # type: ignore
        self.conv.user_query(self.input_ids, is_mm=True)
        self.prompt = self.conv.get_prompt()

        if not os.path.exists(weights_path):
            read_token = '...'
            local_dir = model_path.replace('ermu2001', 'weights')
            snapshot_download(
                model_path,
                local_dir=local_dir,
                repo_type='model',
                local_dir_use_symlinks=True,
                token=read_token,
            )

        self.model, self.processor = load_pllava(
            self.weights_path,
            self.num_frames,
            use_lora=self.use_lora,
            weight_dir=self.weights_dir,
            lora_alpha=self.lora_alpha,
            use_multi_gpus=self.use_multi_gpus
        )  # type: ignore

        if not self.use_multi_gpus:
            self.model = self.model.to(self.device)


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
        video_file, _ = load_video(video_file, num_segments=self.num_segments, return_msg=True, resolution=self.resolution)
        return key, video_file

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()
        keys, video_tensors = list(zip(*batch))
        input_ids_batch = [self.prompt] * len(video_tensors)
        inputs = self.processor(text=input_ids_batch, images=video_tensors, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            output_token = self.model.generate(
                **inputs,
                media_type='video',
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                use_cache = self.use_cache
            )
            output_texts = self.processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        split_tag = self.conv.roles[-1]
        bug_split_tag = "<|im_start|> assistant\n"
        all_outputs: list[Optional[str]] = []
        for output_text in output_texts:
            output_text = output_text.split(split_tag)[-1].split(bug_split_tag)[-1]
            ending = self.conv.sep if isinstance(self.conv.sep, str) else self.conv.sep[1]
            output_text = output_text.removesuffix(ending).strip()
            all_outputs.append(output_text)
        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)
        return df_batch_labels


class Pllava13bFilter(PllavaFilter):
    def __init__(self, **kwargs: Any) -> None:
        model_path: str = 'ermu2001/pllava-13b'
        weights_path: str = 'weights/pllava-13b'
        weights_dir: str = 'weights/pllava-13b'

        super().__init__(model_path=model_path, weights_path=weights_path, weights_dir=weights_dir, prompts=None, **kwargs)

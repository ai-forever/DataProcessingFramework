from io import BytesIO
from typing import Any

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter
import numpy as np
import torch
from decord import VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


prompt_templates = {
    'detailed_video': 'Describe this video and its style in a very detailed manner',
    'short_video': 'Describe this video and its style briefly',
    '1_sentance': "Describe this video very shortly in 1 sentence."
    }
MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


class CogVLM2Filter(VideoFilter):
    """
    CogVLM2 inference class to get captions for auto-labeling videos.
    More info about the model here: https://github.com/THUDM/CogVLM2

    Parameters
    ----------
    prompt: str = '1_sentance'
        Prompt for the model.
    quant: int = 16
        Model quantization mode: 4, 8 or 16
    num_frames: int = 24
        Number of frames to sample from the video
    device: str = "cuda:0"
        Device to use
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """
    def __init__(
        self,
        prompt: str = '1_sentance',
        quant: int = 16,
        num_frames: int = 24,
        temperature: float = 0.05,
        max_new_tokens: int = 1024,
        device: str = "cuda:0",
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.strategy = 'chat'
        self.prompt = prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            # padding_side="left"
        )
        self.num_frames = num_frames

        if quant == 4:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True,
                revision='ca14f13b05f5ead425188aae3e5e725bf4905cd1'
            ).eval()
        elif quant == 8:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True,
                revision='ca14f13b05f5ead425188aae3e5e725bf4905cd1'
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                revision='ca14f13b05f5ead425188aae3e5e725bf4905cd1'
            ).eval().to(device)

        self.query = prompt_templates[prompt]

        self.num_workers = workers
        self.device = device

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @property
    def result_columns(self) -> list[str]:
        return [f"caption_cogvlm"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        video_file = BytesIO(modality2data['video'])
        video_file = self.load_video(video_file, strategy=self.strategy)
        return key, video_file

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        key, video = batch[0]
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=self.query,
            images=[video],
            history=[],
            template_version=self.strategy
        )

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": True,
            "top_p": 0.1,
            "temperature": self.temperature,
            }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        df_batch_labels[self.schema[1]].extend([response])
        df_batch_labels[self.key_column].extend([key])
        return df_batch_labels


    def load_video(self, video_path, strategy='chat'):
        bridge.set_bridge('torch')
        num_frames = self.num_frames

        decord_vr = VideoReader(uri=video_path)
        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            frame_id_list = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

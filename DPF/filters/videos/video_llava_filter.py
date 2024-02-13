from typing import List, Optional, Dict, Union
import torch
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle, conv_templates
from videollava.mm_utils import (KeywordsStoppingCriteria,
                                 get_model_name_from_path,
                                 tokenizer_image_token)
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init

from .video_filter import VideoFilter


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    

class VideoLLaVAFilter(VideoFilter):
    """ 
    Video-LLaVA inference class to get captions for auto-labeling videos.
    More info about the model here: https://github.com/PKU-YuanGroup/Video-LLaVA
    """ 
    def init(self,
             model_name: str = "Video-LLaVA-7B",
             weights_path: str = None, 
             model_base: str = None,
             cache_path: str = "cache_dir",
             conv_mode: str = "llava_v1",
             prompt_templates: Optional[List[str]] = None,
             temperature: float = 0.2,
             max_new_tokens: int = 1024,
             load_4bit: bool = True,
             load_8bit: bool = False,
             device: str = "cuda:0",
             workers: int = 16,
             batch_size: int = 64,
             pbar: bool = True,
            ):
        super().__init__(pbar)
        
        if prompt_templates is None:
            prompt_templates = ["Describe this video in details"]
            
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.user_prompt = prompt_templates
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        disable_torch_init()
        
        pretrainers = load_pretrained_model(weights_path, model_base, model_name,
                                            load_8bit, load_4bit,
                                            device=self.device, cache_dir=cache_path)
        self.tokenizer, self.model, self.processor, self.context_len = pretrainers
        self.video_processor = self.processor['video']
        
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
            
        self.conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            self.roles = ("user", "assistant")
        else:
            self.roles = self.conv.roles
            
        self.schema = [
            self.key_column,
            "caption",
        ]
            
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }
        
    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        video_tensor, special_token = [], []
        video_file = [modality2data['video']]
        video_file = video_processor(video_file, return_tensors='pt')['pixel_values'][0].to(self.model.device,
                                                                                            dtype=torch.float16)
        special_token += [DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames
        video_tensor.append(video_file)
        
        if getattr(self.model.config, "mm_use_im_start_end", False):
            self.user_prompt = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN
                                        for i in special_token]) + '\n' + self.user_prompt[0]
        else:
            self.user_prompt = ''.join(special_token) + '\n' + self.user_prompt[0]
        self.conv.append_message(self.conv.roles[0], self.user_prompt)
        self.conv.append_message(self.conv.roles[1], None)
        self.user_prompt = self.conv.get_prompt()
        
        input_ids = tokenizer_image_token(self.user_prompt, self.tokenizer,
                                          IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else sef.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        return key, video_tensor, input_ids, stopping_criteria
    
    def process(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        for data in batch:
            key, video_tensor, input_ids, stopping_criteria = data
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video_tensor,  # video as fake images
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            caption = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            df_batch_labels[self.key_column].append(key)
            df_batch_labels['caption'].append(caption)
        return df_batch_labels
        
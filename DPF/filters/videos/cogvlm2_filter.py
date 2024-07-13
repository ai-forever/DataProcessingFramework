import torch
from decord import cpu, VideoReader, bridge
from .video_filter import VideoFilter


def load_video(video_bytes, num_frames=24, clip_start_sec=0, clip_end_sec=60):
    bridge.set_bridge('torch')
    decord_vr = VideoReader(video_bytes, ctx=cpu(0))
    total_frames = len(decord_vr)
    start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
    end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
    frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    return video_data.permute(3, 0, 1, 2)


class CogVLM2VideoFilter(VideoFilter):
    """
    CogVLM2-Video inference class to get captions for auto-labeling videos.
    More info about the model here: hhttps://github.com/THUDM/CogVLM2
    """
    def __init__(
            self,
            model_path: str = "THUDM/cogvlm2-video-llama3-chat",
            cache_path: Optional[str] = None,
            prompt: str = "short",
            temperature: float = 0.1,
            max_new_tokens: int = 1024,
            top_k: int = 1,
            top_p: float = 0.1,
            num_frames: int = 24,
            clip_max_sec: float = 60,
            device: str = "cuda:0",
            workers: int = 16,
            batch_size: int = 8,
            pbar: bool = True,
            _pbar_position: int = 0,
    ):
        super().__init__(pbar, _pbar_position)
        self.num_frames = num_frames
        self.clip_max_sec = clip_max_sec
        self.device = device
        self.num_workers = workers
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_path,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval().to(device)

        self.prompt_templates = {
            "short": "Describe this image very shortly in 1-2 short sentences",
        }

        self.gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": 128002,
            "top_k": top_k,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
        }

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
        video_bytes = BytesIO(modality2data['video'])
        video_tensor = load_video(
            video_bytes, 
            num_frames=self.num_frames, 
            clip_end_sec=self.clip_max_sec
        )
        return key, video_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, video_tensors = list(zip(*batch))
        video_tensors = default_collate(video_tensors).to(self.device)  # type: ignore

        history = []

        inputs = self.model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **self.gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        df_batch_labels[self.schema[1]].extend([response])
        df_batch_labels[self.key_column].extend(keys)
        return df_batch_labels

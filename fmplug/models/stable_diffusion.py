# stdlib
from typing import List, Tuple

# third party
import torch
from diffusers import StableDiffusion3Pipeline


class StableDiffusion3Base:
    def __init__(
        self,
        model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
        device='cuda',
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )

        del pipe

    def encode_prompt(
        self, prompt: List[str], batch_size: int = 1
    ) -> Tuple[torch.Tensor, ...]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        text_clip1_emb = self.text_enc_1(
            text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True
        )
        pool_clip1_emb = text_clip1_emb[0].to(
            dtype=self.dtype, device=self.text_enc_1.device
        )
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(
            dtype=self.dtype, device=self.text_enc_1.device
        )

        text_clip2_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        text_clip2_emb = self.text_enc_2(
            text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True
        )
        pool_clip2_emb = text_clip2_emb[0].to(
            dtype=self.dtype, device=self.text_enc_2.device
        )
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(
            dtype=self.dtype, device=self.text_enc_2.device
        )

        # T5 encode (used for text condition)
        text_t5_ids = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt',
        ).input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.text_enc_3.device)

        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb

    def initialize_latent(
        self, img_size: Tuple[int, ...], batch_size: int = 1, **kwargs
    ):
        H, W = img_size
        lH, lW = H // self.vae_scale_factor, W // self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(
            hidden_states=z,
            timestep=t,
            pooled_projections=pooled_emb,
            encoder_hidden_states=prompt_emb,
            return_dict=False,
        )[0]
        return v

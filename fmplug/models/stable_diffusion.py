# stdlib
import inspect
from typing import Any, List, Optional, Tuple, Union

# third party
import torch
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor


class StableDiffusion3Base:
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device="cuda",
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )
        pipe = pipe.to(device)
        self.pipe = pipe

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = pipe.vae
        self.vae.eval()
        # self.vae.requires_grad_(False)
        self.vae.enable_gradient_checkpointing()

        self.transformer = pipe.transformer.to(device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )

        del pipe

    def encode_prompt(
        self, prompt: List[str], batch_size: int = 1
    ) -> Tuple[torch.Tensor, ...]:
        """
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        """
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
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
            return_tensors="pt",
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
            return_tensors="pt",
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


class StableDiffusion3BaseV2:
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        num_inference_steps: int = 3,
        guidance_scale: float = 2.0,
        device="cuda",
        dtype=torch.float32,
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.dtype = dtype

        # We can save more memory by turning off the T5 encoder
        # (text_encoder_3, tokenizer_3)
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_key, text_encoder_3=None, tokenizer_3=None, torch_dtype=self.dtype
        )
        pipe = pipe.to(device)
        self.pipe = pipe

        self.scheduler = pipe.scheduler
        self.prompt_encoder = pipe.encode_prompt
        self.image_processor = pipe.image_processor
        # self.prepare_latents = pipe.prepare_latents

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.enable_gradient_checkpointing()

        self.transformer = pipe.transformer.to(device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()

    def encode_prompts(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        prompt_3: Optional[str] = None,
        negative_prompt: str = "",
        negative_prompt_2: Optional[str] = None,
        negative_prompt_3: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that will encode the prompts for SD3.
        """
        do_classifier_free_guidance = self.guidance_scale > 1.0

        # Defaults from SD3 pipeline
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        clip_skip = None
        num_images_per_prompt = 1
        max_sequence_length = 128
        lora_scale = None

        # encode prompt
        print("Encoding prompts ...")
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.prompt_encoder(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=self.device,
                clip_skip=clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # prompt embeds with classifier free guidance
        if do_classifier_free_guidance:
            print("Classifier free guidance ...")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        return prompt_embeds, pooled_prompt_embeds

    def retrieve_timesteps_and_sigmas(
        self,
        strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Function to collect the timesteps and sigmas for the SD3 pipeline.
        """
        batch_size = 1
        num_images_per_prompt = 1
        scheduler_kwargs: dict[str, Any] = {}
        sigmas = None

        # The sigmas get modified in this step to align with the timesteps
        # and the number of inference steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        timesteps, num_inference_steps = self.pipe.get_timesteps(
            num_inference_steps, strength, self.device
        )
        sigmas = self.pipe.scheduler.sigmas
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        return timesteps, sigmas, latent_timestep

    def prepare_latents(self, latents: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Function to prepare the latent variable based on the strenght.
        TODO: Modify this funtion to match the SD3 pipeline exactly. For now
        we are testing.
        """
        # For image size 512
        input_shape = (1, 16, 64, 64)

        if latents is not None:
            return latents

        latents = randn_tensor(
            input_shape, generator=None, device=self.device, dtype=self.dtype
        )

        return latents

    def predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Function that applies the transformer to inputs to compute the noise
        prediction for SD3.
        """
        result = self.transformer(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        return result


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
            "Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"  # noqa
                f" timestep schedules. Please check whether you are using the correct scheduler."  # noqa
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)  # type: ignore
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"  # noqa
                f" sigmas schedules. Please check whether you are using the correct scheduler."  # noqa
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

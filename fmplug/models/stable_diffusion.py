# stdlib
import inspect
from typing import Any, List, Optional, Tuple, Union

# third party
import torch
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
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


# class InverSD3(StableDiffusion3Img2ImgPipeline):
#     def _get_t5_prompt_embeds(
#         self,
#         prompt: Union[str, List[str]] = None,
#         num_images_per_prompt: int = 1,
#         max_sequence_length: int = 256,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ):
#         device = device or self._execution_device
#         dtype = dtype or self.text_encoder.dtype

#         prompt = [prompt] if isinstance(prompt, str) else prompt
#         batch_size = len(prompt)

#         if self.text_encoder_3 is None:
#             return torch.zeros(
#                 (
#                     batch_size * num_images_per_prompt,
#                     self.tokenizer_max_length,
#                     self.transformer.config.joint_attention_dim,
#                 ),
#                 device=device,
#                 dtype=dtype,
#             )

#         text_inputs = self.tokenizer_3(
#             prompt,
#             padding="max_length",
#             max_length=max_sequence_length,
#             truncation=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )
#         text_input_ids = text_inputs.input_ids
#         untruncated_ids = self.tokenizer_3(
#             prompt, padding="longest", return_tensors="pt"
#         ).input_ids

#         if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
#             text_input_ids, untruncated_ids
#         ):
#             removed_text = self.tokenizer_3.batch_decode(
#                 untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
#             )

#         # prompt_embeds = self.text_encoder_3(
#         #     text_input_ids.to(device), output_hidden_states=True
#         # )

#         # hidden_states = prompt_embeds.hidden_states[0]
#         # print(hidden_states.shape)
#         # print(hidden_states[0, :].shape)

#         # dtype = self.text_encoder_3.dtype
#         # prompt_embeds = prompt_embeds.hidden_states[0].to(dtype=dtype, device=device)

#         prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

#         dtype = self.text_encoder_3.dtype
#         prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

#         _, seq_len, _ = prompt_embeds.shape

#         # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
#         prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#         prompt_embeds = prompt_embeds.view(
#             batch_size * num_images_per_prompt, seq_len, -1
#         )

#         # pooled_prompt_embeds = hidden_states.mean(axis=1)
#         # pooled_prompt_embeds = pooled_prompt_embeds[0, :2048].unsqueeze(0)

#         return prompt_embeds

#     # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
#     def _get_clip_prompt_embeds(
#         self,
#         prompt: Union[str, List[str]],
#         num_images_per_prompt: int = 1,
#         device: Optional[torch.device] = None,
#         clip_skip: Optional[int] = None,
#         clip_model_index: int = 0,
#     ):
#         device = device or self._execution_device

#         clip_tokenizers = [self.tokenizer, self.tokenizer_2]
#         clip_text_encoders = [self.text_encoder, self.text_encoder_2]

#         tokenizer = clip_tokenizers[clip_model_index]
#         text_encoder = clip_text_encoders[clip_model_index]

#         prompt = [prompt] if isinstance(prompt, str) else prompt
#         batch_size = len(prompt)

#         text_inputs = tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=self.tokenizer_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )

#         text_input_ids = text_inputs.input_ids
#         untruncated_ids = tokenizer(
#             prompt, padding="longest", return_tensors="pt"
#         ).input_ids
#         if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
#             text_input_ids, untruncated_ids
#         ):
#             removed_text = tokenizer.batch_decode(
#                 untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
#             )
#         prompt_embeds = text_encoder(
#             text_input_ids.to(device), output_hidden_states=True
#         )

#         print("Text encoded prompt embed shape")
#         print(prompt_embeds[0].shape)

#         pooled_prompt_embeds = prompt_embeds[0]

#         if clip_skip is None:
#             prompt_embeds = prompt_embeds.hidden_states[-2]
#             print(prompt_embeds.shape)
#         else:
#             prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

#         prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

#         _, seq_len, _ = prompt_embeds.shape
#         # duplicate text embeddings for each generation per prompt, using mps friendly method
#         prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#         prompt_embeds = prompt_embeds.view(
#             batch_size * num_images_per_prompt, seq_len, -1
#         )

#         pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
#         pooled_prompt_embeds = pooled_prompt_embeds.view(
#             batch_size * num_images_per_prompt, -1
#         )

#         return prompt_embeds, pooled_prompt_embeds

#     def encode_prompt(
#         self,
#         prompt: Union[str, List[str]],
#         prompt_2: Union[str, List[str]],
#         prompt_3: Union[str, List[str]],
#         device: Optional[torch.device] = None,
#         num_images_per_prompt: int = 1,
#         do_classifier_free_guidance: bool = True,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         negative_prompt_2: Optional[Union[str, List[str]]] = None,
#         negative_prompt_3: Optional[Union[str, List[str]]] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         clip_skip: Optional[int] = None,
#         max_sequence_length: int = 256,
#         lora_scale: Optional[float] = None,
#     ):
#         device = device or self._execution_device

#         prompt = [prompt] if isinstance(prompt, str) else prompt
#         if prompt is not None:
#             batch_size = len(prompt)
#         else:
#             batch_size = prompt_embeds.shape[0]

#         if prompt_embeds is None:
#             prompt_2 = prompt_2 or prompt
#             prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

#             prompt_3 = prompt_3 or prompt
#             prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

#             prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
#                 prompt=prompt,
#                 device=device,
#                 num_images_per_prompt=num_images_per_prompt,
#                 clip_skip=clip_skip,
#                 clip_model_index=0,
#             )
#             print(prompt_embed.shape)
#             print(pooled_prompt_embed.shape)

#             prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
#                 prompt=prompt_2,
#                 device=device,
#                 num_images_per_prompt=num_images_per_prompt,
#                 clip_skip=clip_skip,
#                 clip_model_index=1,
#             )
#             clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
#             clip_prompt_embeds = prompt_embed

#             print(max_sequence_length)
#             t5_prompt_embed = self._get_t5_prompt_embeds(
#                 prompt=prompt_3,
#                 num_images_per_prompt=num_images_per_prompt,
#                 max_sequence_length=max_sequence_length,
#                 device=device,
#             )

#             clip_prompt_embeds = torch.nn.functional.pad(
#                 clip_prompt_embeds,
#                 (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
#             )

#             prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
#             pooled_prompt_embeds = torch.cat(
#                 [pooled_prompt_embed, pooled_prompt_2_embed], dim=-1
#             )
#             print("T5 prompt embed shape")
#             print(t5_prompt_embed.shape)

#         if do_classifier_free_guidance and negative_prompt_embeds is None:
#             negative_prompt = negative_prompt or ""
#             negative_prompt_2 = negative_prompt_2 or negative_prompt
#             negative_prompt_3 = negative_prompt_3 or negative_prompt

#             # normalize str to list
#             negative_prompt = (
#                 batch_size * [negative_prompt]
#                 if isinstance(negative_prompt, str)
#                 else negative_prompt
#             )
#             negative_prompt_2 = (
#                 batch_size * [negative_prompt_2]
#                 if isinstance(negative_prompt_2, str)
#                 else negative_prompt_2
#             )
#             negative_prompt_3 = (
#                 batch_size * [negative_prompt_3]
#                 if isinstance(negative_prompt_3, str)
#                 else negative_prompt_3
#             )

#             if prompt is not None and type(prompt) is not type(negative_prompt):
#                 raise TypeError(
#                     f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
#                     f" {type(prompt)}."
#                 )
#             elif batch_size != len(negative_prompt):
#                 raise ValueError(
#                     f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
#                     f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
#                     " the batch size of `prompt`."
#                 )

#             negative_prompt_embed, negative_pooled_prompt_embed = (
#                 self._get_clip_prompt_embeds(
#                     negative_prompt,
#                     device=device,
#                     num_images_per_prompt=num_images_per_prompt,
#                     clip_skip=None,
#                     clip_model_index=0,
#                 )
#             )
#             negative_prompt_2_embed, negative_pooled_prompt_2_embed = (
#                 self._get_clip_prompt_embeds(
#                     negative_prompt_2,
#                     device=device,
#                     num_images_per_prompt=num_images_per_prompt,
#                     clip_skip=None,
#                     clip_model_index=1,
#                 )
#             )
#             # negative_clip_prompt_embeds = torch.cat(
#             #     [negative_prompt_embed, negative_prompt_2_embed], dim=-1
#             # )
#             negative_clip_prompt_embeds = negative_prompt_embed

#             t5_negative_prompt_embed = self._get_t5_prompt_embeds(
#                 prompt=negative_prompt_3,
#                 num_images_per_prompt=num_images_per_prompt,
#                 max_sequence_length=max_sequence_length,
#                 device=device,
#             )

#             negative_clip_prompt_embeds = torch.nn.functional.pad(
#                 negative_clip_prompt_embeds,
#                 (
#                     0,
#                     t5_negative_prompt_embed.shape[-1]
#                     - negative_clip_prompt_embeds.shape[-1],
#                 ),
#             )

#             negative_prompt_embeds = torch.cat(
#                 [negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2
#             )
#             negative_pooled_prompt_embeds = torch.cat(
#                 [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
#             )

#             # print("Final Output")
#             # print(prompt_embeds.shape)
#             # print(pooled_prompt_embeds.shape)

#         return (
#             # prompt_embeds,
#             # negative_prompt_embeds,
#             t5_prompt_embed,
#             t5_negative_prompt_embed,
#             pooled_prompt_embeds,
#             negative_pooled_prompt_embeds,
#         )


class StableDiffusion3BaseV2:
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        scheduler: Optional[FlowMatchHeunDiscreteScheduler] = None,
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
        if scheduler is None:
            pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
                model_key, text_encoder_3=None, tokenizer_3=None, torch_dtype=self.dtype
            )

            # pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            #     model_key, torch_dtype=self.dtype
            # )

        else:
            pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
                model_key,
                scheduler=scheduler,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=self.dtype,
            )

        pipe = pipe.to(device)
        self.pipe = pipe
        self.pipe.tokenizer_max_length = 77

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
        clip_skip = 4
        num_images_per_prompt = 1
        max_sequence_length = 256
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

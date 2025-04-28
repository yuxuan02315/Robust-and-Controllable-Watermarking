from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler

from modified_stable_diffusion import ModifiedStableDiffusionPipeline


### credit to: https://github.com/cccntu/efficient-prompt-to-prompt

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
            alpha_tm1 ** 0.5
            * (
                    (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
                    + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler,
                 safety_checker,
                 feature_extractor,
                 requires_safety_checker: bool = True,
                 ):
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                                                                text_encoder,
                                                                tokenizer,
                                                                unet,
                                                                scheduler,
                                                                safety_checker,
                                                                feature_extractor,
                                                                requires_safety_checker)

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    """这个方法通常用于图像生成过程的第一步，即生成一组随机的潜在空间数据,
    这些数据随后将通过 U-Net 网络进行处理，以生成最终的图像。"""
    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    """这个方法首先使用分词器将文本提示编码为数字ID，然后使用文本编码器将这些ID转换为嵌入向量，
    这些嵌入向量可以捕捉文本内容的高级语义信息，并用于指导稳定扩散模型生成图像。"""
    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    """这个方法通常用于将现有的图像转换为其在潜在空间中的表示"""
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        # 使用变分自编码器（VAE）对图像进行编码，得到潜在空间的分布
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def backward_diffusion(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            edge_features: Optional[torch.FloatTensor] = None,  # 新增参数
            **kwargs,
    ):
        """ Generate image from text prompt and latents该方法用于从文本提示和潜在空间(latents)生成图像
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual  初始化约束
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # ddim 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
            # # 融合边缘特征
            # # 在融合边缘特征时使用加权平均
            # if edge_features is not None:
            #     # 归一化边缘特征
            #     # edge_features = (edge_features - edge_features.min()) / (edge_features.max() - edge_features.min())
            #     edge_features = (edge_features - edge_features.mean()) / (edge_features.std() + 1e-8)
            #     # 使用非线性变换融合
            #     scale_factor = 0.001  # 根据需要调整缩放因子
            #     # latents = latents * (1 - scale_factor) + torch.sigmoid(edge_features) * scale_factor
            #     # latents = latents + torch.sigmoid(edge_features) * scale_factor
            #     latents = latents + edge_features * scale_factor
            #     # print(f"Fusing edge features at step {i}, shape: {edge_features.shape}")
            # 融合边缘特征
            if edge_features is not None:
                # 归一化边缘特征（零均值单位方差）
                edge_features = (edge_features - edge_features.mean()) / (edge_features.std() + 1e-8)

                # 动态缩放因子计算（保持张量类型）
                T_tensor = torch.tensor(num_inference_steps, device=edge_features.device)
                current_step_tensor = torch.tensor(i, device=edge_features.device)
                t = T_tensor - current_step_tensor  # 张量运算

                # 相位敏感型权重调整（动态衰减，早期大后期小）
                if t > 2 * T_tensor / 3:
                    # 高斯衰减：初始值0.0005，随剩余步数减少逐步衰减
                    sigma1 = T_tensor / 3
                    exponent = -((T_tensor - t) ** 2) / (2 * (sigma1 ** 2))
                    scale_factor = 0.0005 * torch.exp(exponent)
                elif T_tensor / 3 < t <= 2 * T_tensor / 3:
                    # 线性过渡（确保衔接高斯衰减末端值）
                    start_value = 0.0005 * torch.exp(torch.tensor(-0.5, device=T_tensor.device))  # 使用张量
                    end_value = torch.tensor(0.0001, device=T_tensor.device)  # 显式定义设备
                    ratio = (2 * T_tensor / 3 - t) / (T_tensor / 3)
                    scale_factor = start_value + (end_value - start_value) * ratio
                else:
                    # 指数衰减到基线值0.0001
                    sigma2 = 0.0001 * T_tensor
                    exponent = -t / sigma2
                    scale_factor = 0.0001 + 0.0001 * torch.exp(exponent)

                # 确保缩放因子与潜在变量设备一致
                scale_factor = scale_factor.to(latents.dtype).to(latents.device)

                # 时域自适应融合
                latents = latents + torch.sigmoid(edge_features) * scale_factor

        return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i: i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

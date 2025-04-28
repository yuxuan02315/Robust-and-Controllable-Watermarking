import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy


# 读取一个 JSON 格式的数据文件，并将其解析为适合 Python 使用的字典格式。

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),  # 将图像的大小调整为指定的目标尺寸。
            transforms.CenterCrop(target_size),  # 对已经调整大小的图像进行中心裁剪，确保输出图像为正方形且大小为 target_size
            transforms.ToTensor(),  # 将图像转换为 PyTorch 张量,并将像素值从 [0, 255] 范围线性映射到 [0, 1] 范围。
        ]
    )
    image = tform(image)  # 将定义的转换应用于输入的 image
    # 将图像的像素值从 [0, 1] 范围调整为 [-1, 1] 范围
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    # 使用管道(pipe)的解码功能将潜在空间数据(latents)解码为图像
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


"""该函数用于对两张图像（img1 和 img2）进行一系列的图像扭曲操作"""


def image_distortion(img1, img2, seed, args):
    # 如果设置了随机旋转的角度，则对两张图像进行随机旋转
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)
    # 如果设置了JPEG压缩比，则对两张图像进行JPEG压缩
    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
    # 如果设置了裁剪比例和裁剪尺度，则对两张图像进行随机裁剪
    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale),
                                            ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale),
                                            ratio=(args.crop_ratio, args.crop_ratio))(img2)
    # 如果设置了高斯模糊的半径，则对两张图像进行高斯模糊
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
    # 如果设置了高斯噪声的标准差，则对两张图像添加高斯噪声
    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
    # 如果设置了亮度因子，则对两张图像进行亮度调整
    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images用于计算图像和文本之间的相似度
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    # 使用 with torch.no_grad() 上下文管理器，以避免计算梯度，从而提高效率
    with torch.no_grad():
        # 将所有图像转换为 PyTorch 张量，并将其堆叠成一个批次
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        # 使用模型对图像进行编码，得到图像特征
        image_features = model.encode_image(img_batch)

        # 使用分词器将提示文本转换为 PyTorch 张量，并将其堆叠成一个批次
        text = tokenizer([prompt]).to(device)
        # 使用模型对文本进行编码，得到文本特征
        text_features = model.encode_text(text)
        # 标准化图像特征和文本特征
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 使用点积计算图像特征和文本特征之间的相似度
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    # 检查数据集名称是否包含 'laion'
    if 'laion' in args.dataset:
        # 如果包含 'laion'，则加载数据集的 'train' 部分
        dataset = load_dataset(args.dataset)['train']
        # 设置 prompt_key 为 'TEXT'
        prompt_key = 'TEXT'
    # 检查数据集名称是否包含 'coco'
    elif 'coco' in args.dataset:
        # # 如果包含 'coco'，则加载数据集的元数据
        # with open('fid_outputs/coco/meta_data.json') as f:
        #     dataset = json.load(f)
        #     dataset = dataset['annotations']
        #     prompt_key = 'caption'
        # 使用HuggingFace官方数据集加载方式
        from datasets import load_dataset

        # 加载MS-COCO-2017数据集
        coco_dataset = load_dataset('mscoco', '2017', split=f'{args.coco_split}2017')

        # 处理annotations获取caption
        def process_coco(examples):
            captions = []
            for anns in examples['annotations']:
                # 获取指定索引的caption
                if len(anns) > args.caption_index:
                    captions.append(anns[args.caption_index]['caption'])
                else:
                    captions.append("")
            return {'prompt': captions}

        dataset = coco_dataset.map(process_coco, batched=True)
        dataset = dataset.filter(lambda x: x['prompt'] != "")  # 过滤空prompt
        prompt_key = 'prompt'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


"""用于生成一个中心在 (x0, y0) 位置、半径为 r 的圆形遮罩"""


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    # 创建一个大小为 size 的网格，网格的行（y轴）被反转，以便遮罩可以正确地应用到图像上
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def get_watermarking_mask(init_latents_w, args, device):
    # 创建一个初始潜在空间数据大小的全零布尔张量，用于水印遮罩
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        # 调用之前定义的 circle_mask 函数，生成一个圆形遮罩
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        # 将生成的 numpy 遮罩转换为 PyTorch 张量，并移动到指定的设备上
        torch_mask = torch.tensor(np_mask).to(device)

        # 如果指定了水印通道，则仅对该通道应用遮罩
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        # 获取潜在空间数据在最后一个维度上的中心点
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p - args.w_radius:anchor_p + args.w_radius,
            anchor_p - args.w_radius:anchor_p + args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p - args.w_radius:anchor_p + args.w_radius,
            anchor_p - args.w_radius:anchor_p + args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        # 如果水印遮罩形状参数不支持，则抛出 NotImplementedError
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


"""这个函数主要用于生成不同类型的水印模式，这些模式可以用于图像处理或水印嵌入"""


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


"""这个函数主要用于将水印嵌入到图像的初始潜伏变量中。
通过选择不同的水印嵌入方式，可以在图像的不同域（例如，空间域或频域）中嵌入水印。
"""


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(
            reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[
        watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[
        watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])

    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w

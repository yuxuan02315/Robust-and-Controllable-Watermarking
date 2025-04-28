import cv2
import os
import numpy as np

# 设置图片和保存路径
input_dir = '/tree_ring_watermarking/tree-ring-watermark-main/generated_no_w_images'  # 存储原始图片的文件夹
output_dir = '/tree_ring_watermarking/tree-ring-watermark-main/edge_images'  # 存储素描图片的文件夹
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，则创建文件夹

# 获取文件夹中的所有图片
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# 提取精确素描并保存
for image_file in image_files:
    # 构造图片路径
    image_path = os.path.join(input_dir, image_file)

    # 读取图片
    img = cv2.imread(image_path)

    # 将图片转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(blurred_img, 50, 150)

    # 反转边缘图像，使其符合素描效果
    inverted_edges = cv2.bitwise_not(edges)

    # 保存反转后的边缘图像
    edge_image_path = os.path.join(output_dir, f"edge_{image_file}")
    cv2.imwrite(edge_image_path, inverted_edges)

    print(f"Saved edge of {image_file} to {edge_image_path}")
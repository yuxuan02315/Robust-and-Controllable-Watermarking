import cv2
import os
import numpy as np
import mediapipe as mp

# 设置路径（保持与之前相同）
input_dir = '/tree_ring_watermarking/tree-ring-watermark-main/generated_no_w_images'
output_dir = '/tree_ring_watermarking/tree-ring-watermark-main/pose_results'
os.makedirs(output_dir, exist_ok=True)

# 初始化MediaPipe姿势模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,  # 单张图片模式
    model_complexity=2,  # 模型复杂度（0-2）
    enable_segmentation=False,  # 不需要分割蒙版
    min_detection_confidence=0.5
)

# 处理图像文件
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(input_dir, img_file)
    output_path = os.path.join(output_dir, f"pose_{img_file}")

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue

    # 转换颜色空间 BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 执行推理
    results = pose.process(img_rgb)

    # 可视化结果
    annotated_img = img.copy()
    if results.pose_landmarks:
        # 绘制关键点
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2)
        )

        # 保存关键点坐标
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x * img.shape[1],  # 转换为绝对坐标
                              landmark.y * img.shape[0],
                              landmark.visibility])
        np.save(os.path.join(output_dir, f"kpts_{os.path.splitext(img_file)[0]}.npy"),
                np.array(keypoints))

    # 保存结果图
    cv2.imwrite(output_path, annotated_img)
    print(f"Processed {img_file} -> Saved to {output_path}")

# 释放资源
pose.close()
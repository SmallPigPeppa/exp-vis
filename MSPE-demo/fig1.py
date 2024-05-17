from PIL import Image
import os

def split_and_reassemble(image_path, output_dir_patches, output_dir_reassembled, num_rows, num_cols, spacing=10):
    # 确保输出目录存在
    os.makedirs(output_dir_patches, exist_ok=True)
    os.makedirs(output_dir_reassembled, exist_ok=True)

    # 打开图片并获取尺寸
    img = Image.open(image_path)
    width, height = img.size

    # 计算每个patch的尺寸
    patch_width = width // num_cols
    patch_height = height // num_rows

    patches = []

    # 切割图片并保存
    for i in range(num_rows):
        for j in range(num_cols):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height

            patch = img.crop((left, upper, right, lower))
            patch_path = os.path.join(output_dir_patches, f'patch_{i * num_cols + j}.jpg')
            patch.save(patch_path)
            patches.append(patch)

    # 创建新的空白图用于拼接patches
    new_width = width + (num_cols - 1) * spacing
    new_height = height + (num_rows - 1) * spacing
    new_img = Image.new('RGB', (new_width, new_height), "white")

    # 拼接patches
    y_offset = 0
    for i in range(num_rows):
        x_offset = 0
        for j in range(num_cols):
            new_img.paste(patches[i * num_cols + j], (x_offset, y_offset))
            x_offset += patch_width + spacing
        y_offset += patch_height + spacing

    # 保存拼接后的图片
    reassembled_image_path = os.path.join(output_dir_reassembled, 'reassembled.jpg')
    new_img.save(reassembled_image_path)

# 使用示例
split_and_reassemble('demo1.jpg', 'output_patches', 'output_reassembled', 3, 3, spacing=100)

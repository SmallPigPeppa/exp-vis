from PIL import Image
import os

def resize_image_to_uniform_shorter_side(img, target_size=1024):
    width, height = img.size
    if width < height:
        scale = target_size / width
        new_size = (target_size, int(height * scale))
    else:
        scale = target_size / height
        new_size = (int(width * scale), target_size)
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)  # 更改这里
    return resized_img

def split_and_reassemble(image_path, output_dir_patches, output_dir_reassembled, num_rows, num_cols, spacing=10, target_size=512):
    os.makedirs(output_dir_patches, exist_ok=True)
    os.makedirs(output_dir_reassembled, exist_ok=True)

    img = Image.open(image_path)
    img = resize_image_to_uniform_shorter_side(img, target_size=target_size)
    width, height = img.size

    patch_width = width // num_cols
    patch_height = height // num_rows
    patches = []

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

    new_width = width + (num_cols - 1) * spacing
    new_height = height + (num_rows - 1) * spacing
    new_img = Image.new('RGB', (new_width, new_height), "white")

    y_offset = 0
    for i in range(num_rows):
        x_offset = 0
        for j in range(num_cols):
            new_img.paste(patches[i * num_cols + j], (x_offset, y_offset))
            x_offset += patch_width + spacing
        y_offset += patch_height + spacing

    reassembled_image_path = os.path.join(output_dir_reassembled, 'reassembled.jpg')
    new_img.save(reassembled_image_path)

# 使用示例
# split_and_reassemble('demo1.jpg', 'output_patches', 'output_reassembled', 3, 3, spacing=50)
# split_and_reassemble('demo2.jpg', 'output_patches', 'output_reassembled', 3, 3, spacing=50)
split_and_reassemble('demo3.jpg', 'output_patches', 'output_reassembled', 3, 3, spacing=50)

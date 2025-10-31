import os
from PIL import Image

# === 配置部分 ===
# 输入和输出文件夹路径
input_folder = r"E:\Py_projects\Patent\Code\Cut\Cut_in"     # 原图文件夹
output_folder = r"E:\Py_projects\Patent\Code\Cut\Cut_out"   # 输出文件夹
mode = "horizontal"   # 可选: "vertical"（纵向） 或 "horizontal"（横向）

# 若输出文件夹不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 定义命名后缀（从左到右 / 从上到下顺序）
suffixes = ['_c', '_s', '_o', '_e']

# === 主处理逻辑 ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        img = Image.open(input_path)
        width, height = img.size
        base_name = os.path.splitext(filename)[0]

        if mode == "vertical":
            # 纵向裁剪（上下切成四份）
            part_height = height // 4
            for i in range(4):
                top = i * part_height
                bottom = (i + 1) * part_height if i < 3 else height
                cropped = img.crop((0, top, width, bottom))
                output_name = f"{base_name}{suffixes[i]}.png"
                output_path = os.path.join(output_folder, output_name)
                cropped.save(output_path)
                print(f"纵向裁剪保存：{output_name}")

        elif mode == "horizontal":
            # 横向裁剪（左右切成四份）
            part_width = width // 4
            for i in range(4):
                left = i * part_width
                right = (i + 1) * part_width if i < 3 else width
                cropped = img.crop((left, 0, right, height))
                output_name = f"{base_name}{suffixes[i]}.png"
                output_path = os.path.join(output_folder, output_name)
                cropped.save(output_path)
                print(f"横向裁剪保存：{output_name}")

print("✅ 所有图片裁剪完成！")

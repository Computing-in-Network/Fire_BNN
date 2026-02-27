import os
import shutil
from PIL import Image

# ==========================================================
# 配置中心：在此统一管理地址和参数
# ==========================================================
# 初始大图存放地址
INPUT_PATH   = os.path.expanduser("./Fire_BNN_Tool_0227/figure/") 
# 所有切块小图存放的统一地址
OUT_BASE_DIR = os.path.expanduser("./Fire_BNN_Tool_0227/test_images")

TILE_SIZE = 64
STRIDE    = 64
PAD       = True
FMT       = "jpg"
# 指定要处理的图片文件名（放在 INPUT_PATH 目录下）。
# 设为 None 或空字符串则处理目录中所有图片。
TARGET_IMAGE = "test1.jpg"  # 示例: "test1.jpg" 或 "test2.jpg" 或 None
# ==========================================================

def tile_image(img: Image.Image, outdir: str, prefix: str, tile=32, stride=32, pad=True, fmt="jpg"):
    """
    prefix: 大图的文件名编号（如 '1', '2' 等）
    """
    os.makedirs(outdir, exist_ok=True)
    img = img.convert("RGB")
    w, h = img.size

    # Padding 逻辑：确保边缘也能整齐切块
    if pad:
        new_w = ((w + tile - 1) // tile) * tile
        new_h = ((h + tile - 1) // tile) * tile
        if (new_w, new_h) != (w, h):
            canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            canvas.paste(img, (0, 0))
            img = canvas
            w, h = img.size

    idx = 0
    metadata = []

    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            patch = img.crop((x, y, x + tile, y + tile))
            
            # --- 文件名重新加入坐标信息以适配发送脚本的正则解析 ---
            name = f"{prefix}_patch_{idx:04d}_x{x}_y{y}.{fmt}"
            
            patch.save(os.path.join(outdir, name), quality=95)
            metadata.append(f"name={name}, x={x}, y={y}")
            idx += 1

    # 生成对应的 Manifest 文件记录元数据
    manifest_name = f"{prefix}_manifest.txt"
    with open(os.path.join(outdir, manifest_name), "w") as f:
        f.write(f"orig_img={prefix}, w={w}, h={h}, tile={tile}, stride={stride}, count={idx}\n")
        f.write("\n".join(metadata))
    
    print(f"完成：大图 [{prefix}] -> 生成 {idx} 个切块。")

def main():
    # 1. 检查输入路径
    if not os.path.exists(INPUT_PATH):
        print(f"错误：找不到输入目录 {INPUT_PATH}")
        return

    # 2. 清空输出目录
    if os.path.exists(OUT_BASE_DIR):
        print(f"正在清空输出目录: {OUT_BASE_DIR} ...")
        # 遍历删除目录下的所有内容，但保留目录本身
        for filename in os.listdir(OUT_BASE_DIR):
            file_path = os.path.join(OUT_BASE_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")
    else:
        os.makedirs(OUT_BASE_DIR)

    # 3. 获取并排序所有图片
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(valid_exts)]
    files.sort() 

    if not files:
        print("提示：没有找到可处理的图片。")
        return
    # 如果在初始化处指定了 TARGET_IMAGE，则只处理指定文件
    if TARGET_IMAGE and TARGET_IMAGE.strip():
        if TARGET_IMAGE not in files:
            print(f"错误：指定的 TARGET_IMAGE={TARGET_IMAGE} 不存在于 {INPUT_PATH}")
            return
        print(f"仅处理初始化指定的图片: {TARGET_IMAGE}")
        f = TARGET_IMAGE
        fpath = os.path.join(INPUT_PATH, f)
        img_id = os.path.splitext(f)[0]
        with Image.open(fpath) as img:
            tile_image(
                img,
                OUT_BASE_DIR,
                prefix=img_id,
                tile=TILE_SIZE,
                stride=STRIDE,
                pad=PAD,
                fmt=FMT,
            )
        return

    print(f"开始处理，所有小图将存入: {OUT_BASE_DIR}")
    
    for f in files:
        fpath = os.path.join(INPUT_PATH, f)
        img_id = os.path.splitext(f)[0]
        
        with Image.open(fpath) as img:
            tile_image(
                img, 
                OUT_BASE_DIR, 
                prefix=img_id, 
                tile=TILE_SIZE, 
                stride=STRIDE, 
                pad=PAD, 
                fmt=FMT
            )

if __name__ == "__main__":
    main()

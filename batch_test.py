import ctypes
import os
import glob
import numpy as np
from PIL import Image
import time

# --- 配置 ---
SO_LIB_PATH = './libfire_engine.so'
MODEL_PATH = './resnet18_clean.ml'
IMAGE_FOLDER = './test_images'  # 你的图片文件夹路径
OUTPUT_FILE = 'result.txt'      # 结果输出路径

# --- 1. 加载 C 动态库 ---
if not os.path.exists(SO_LIB_PATH):
    print(f"Error: {SO_LIB_PATH} not found! Did you compile it?")
    exit(1)

lib = ctypes.CDLL(SO_LIB_PATH)

# 定义函数签名
lib.init_engine.argtypes = [ctypes.c_char_p]
lib.init_engine.restype = ctypes.c_int

lib.predict_image.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.predict_image.restype = ctypes.POINTER(ctypes.c_float)

lib.free_engine.argtypes = []
lib.free_engine.restype = None

# --- 2. 辅助函数 ---

def softmax(x, temperature=1.0):
    """
    temperature: 温度系数。
    BNN 输出数值很大(约2000~3000)，必须使用大 Temperature (如 2000.0) 才能得到正常的概率分布。
    """
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x)) # 减去max防止溢出
    return e_x / e_x.sum()

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32)
        arr = arr.transpose(2, 0, 1) # HWC -> CHW
        arr = (arr / 127.5) - 1.0    # Normalize
        data = arr.flatten().astype(np.float32)
        return np.ascontiguousarray(data)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

# --- 3. 主流程 ---

def main():
    # 初始化
    print(f"[Python] Loading model from {MODEL_PATH}...")
    ret = lib.init_engine(MODEL_PATH.encode('utf-8'))
    if ret != 0:
        print("Failed to initialize engine!")
        return

    # 找图
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    
    image_files.sort()
    print(f"[Python] Found {len(image_files)} images.")

    # 打开输出文件
    with open(OUTPUT_FILE, 'w') as f_out:
        print(f"[Python] Running inference, writing to {OUTPUT_FILE}...")
        
        for img_path in image_files:
            # 1. 预处理
            img_data = preprocess_image(img_path)
            if img_data is None: continue
            
            # 2. 获取指针
            c_float_ptr = img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # 3. 推理
            result_ptr = lib.predict_image(c_float_ptr)
            
            # 4. 解析结果
            logits = [result_ptr[0], result_ptr[1]]
            
            # 关键：设置温度系数为 2000.0 以平滑 BNN 的大数值输出
            probs = softmax(logits, temperature=2000.0)
            
            safe_prob = probs[0]
            fire_prob = probs[1]
            
            # 判定标签
            label = "FIRE" if fire_prob > 0.5 else "SAFE"
            
            filename = os.path.basename(img_path)
            
            # 5. 格式化输出字符串 (使用制表符 \t 或空格对齐)
            # 格式：文件名    class0(safe): 0.xxxx    class1(Fire): 0.xxxx    Score= 0.xxxx (FIRE or SAFE)
            output_line = (
                f"{filename:<20} "                # 文件名左对齐，占20字符
                f"class0(safe): {safe_prob:.6f}    "
                f"class1(Fire): {fire_prob:.6f}    "
                f"Score={fire_prob:.6f} ({label})"
            )
            
            # 写入文件
            f_out.write(output_line + "\n")
            
            # 打印到控制台 (可选)
            print(output_line)

    lib.free_engine()
    print(f"Done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
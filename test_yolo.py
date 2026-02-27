import os
import cv2
from ultralytics import YOLO

# --- 配置参数 ---
# 使用你接收端程序中定义的模型路径
MODEL_PATH = "/Fire_r/app/models/fire.pt" 
# 输入图片文件夹
INPUT_FOLDER = "figure"
# 结果保存文件夹
OUTPUT_FOLDER = "test_results"
# 检测置信度阈值
CONF_THRESHOLD = 0.25

def run_test():
    # 1. 检查并创建文件夹
    if not os.path.exists(INPUT_FOLDER):
        print(f"[错误] 找不到输入文件夹: {INPUT_FOLDER}")
        return
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. 加载模型
    print(f"[初始化] 正在加载模型: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    # 3. 遍历并检测图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"[提示] 在 {INPUT_FOLDER} 中没有找到图片文件。")
        return

    print(f"[启动] 开始处理 {len(image_files)} 张图片...")

    for img_name in image_files:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        
        # 运行推理
        # verbose=False 减少日志输出，device='cpu' 或 '0' (如果有GPU)
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        
        # 结果处理
        result = results[0]
        
        # 使用 YOLO 自带的 plot() 函数获取画好框的图片 (BGR 格式)
        annotated_frame = result.plot()

        # 保存图片
        save_path = os.path.join(OUTPUT_FOLDER, f"res_{img_name}")
        cv2.imwrite(save_path, annotated_frame)
        
        # 打印检测信息
        boxes = result.boxes
        print(f"  - 图片: {img_name} | 检测到目标数: {len(boxes)}")

    print(f"\n[完成] 所有结果已保存至: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_test()
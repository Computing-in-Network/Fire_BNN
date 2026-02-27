import numpy as np
import socket
import struct
import cv2
import time
import csv
import os
from ultralytics import YOLO

# --- 配置 (功能的开关放在此处) ---
MODEL_PATH = "/home/yhli/yolo/yolo_projects/fire_mission/runs4/weights/best.pt"
CONF_THRESHOLD = 0.25
CHECK_INTERVAL = 25 
STABLE_IOU_THR = 0.90
FINAL_CONF_THR = 0.70
STABLE_PATIENCE = 5 
LISTEN_PORT = 5005 

# --- 新增开关 ---
SAVE_EVERY_CHECK = True  # 是否在每次推理决策时都保存图片（用于分析震荡）

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

class ReceiverEngine:
    def __init__(self):
        print(f"[初始化] 加载模型: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self.canvas = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        self.history = None
        self.stop_inference = False
        self.csv_file = None
        self.csv_writer = None
        self.current_task_id = -1
        self.final_saved = False
        
        # 确保 result 文件夹存在
        os.makedirs("result", exist_ok=True)
        
        # 用于裁剪空白区域的边界追踪
        self.content_bounds = [float('inf'), float('inf'), 0, 0] # min_x, min_y, max_x, max_y

    def reset(self, task_id):
        self.canvas.fill(128)
        self.history = None
        self.stop_inference = False
        self.final_saved = False
        self.current_task_id = task_id
        self.content_bounds = [float('inf'), float('inf'), 0, 0]
        
        # 数据文件保存在 result 文件夹
        csv_filename = os.path.join("result", f"task_{task_id}_metrics.csv")
        file_exists = os.path.isfile(csv_filename)
        self.csv_file = open(csv_filename, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow(["Sequence", "Trans_Ratio", "IoU", "Conf", "Smoothed", "Stable", "Time_ms"])
        
        print("\n" + "="*40)
        print(f"[接收端] 开始新任务。任务ID: {task_id}")

    def update_bounds(self, x, y, tile_size):
        self.content_bounds[0] = min(self.content_bounds[0], x)
        self.content_bounds[1] = min(self.content_bounds[1], y)
        self.content_bounds[2] = max(self.content_bounds[2], x + tile_size)
        self.content_bounds[3] = max(self.content_bounds[3], y + tile_size)

    def decode_nv12_to_bgr(self, data, tile):
        if not data: return None
        try:
            nv12_np = np.frombuffer(data, dtype=np.uint8).reshape((tile * 3 // 2, tile))
            return cv2.cvtColor(nv12_np, cv2.COLOR_YUV2BGR_NV12)
        except Exception:
            return None

    def sccs_check(self, current):
        if current is None: return False, 0
        if self.history is None:
            current['stable_count'] = 0
            self.history = current
            return False, 0
        
        iou_val = calculate_iou(current['box'], self.history['box'])
        stable_count = self.history['stable_count'] + 1 if iou_val > STABLE_IOU_THR else 0
        current['stable_count'] = stable_count
        self.history = current

        if current['score'] > FINAL_CONF_THR and stable_count >= STABLE_PATIENCE:
            return True, iou_val
        return False, iou_val

    def save_result(self, seq, is_final=False):
        """根据有效内容区域裁剪并保存"""
        pad = 10
        x1 = max(0, int(self.content_bounds[0]) - pad)
        y1 = max(0, int(self.content_bounds[1]) - pad)
        x2 = min(self.canvas.shape[1], int(self.content_bounds[2]) + pad)
        y2 = min(self.canvas.shape[0], int(self.content_bounds[3]) + pad)

        if x1 >= x2 or y1 >= y2: return

        res_canvas = self.canvas[y1:y2, x1:x2].copy()

        if self.history:
            b = self.history['box'].astype(int)
            b_rel = [b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1]
            cv2.rectangle(res_canvas, (b_rel[0], b_rel[1]), (b_rel[2], b_rel[3]), (0, 0, 255), 2)
            cv2.putText(res_canvas, f"Fire: {self.history['score']:.2f}", (b_rel[0], b_rel[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 后缀区分：如果是最终决策则加个标记
            suffix = "_final" if is_final else ""
            save_name = os.path.join("result", f"task_{self.current_task_id}_seq_{seq}{suffix}.jpg")
            cv2.imwrite(save_name, res_canvas)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", LISTEN_PORT))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024) 
        
        print(f"[接收端] 正在监听 ALF 格式包 (端口 {LISTEN_PORT})...")
        packet_count = 0

        while True:
            packet, addr = sock.recvfrom(2048)
            if len(packet) < 832: continue 

            header_data = packet[:35]
            try:
                task_id, op, dt, v_len, tile_len, total_pkts, \
                x1, y1, x2, y2, vaddr1, timestamp_ms, flag = struct.unpack("<HBBHHHHHHHQQB", header_data)
            except Exception as e:
                print(f"[报头解析错误]: {e}")
                continue

            if task_id != self.current_task_id:
                if self.csv_file: self.csv_file.close()
                self.reset(task_id)
                packet_count = 0

            try:
                raw1 = packet[64 : 64 + tile_len]
                raw2 = packet[64 + tile_len : 64 + 2 * tile_len]
                tile_size = 16 
                self.update_bounds(x1, y1, tile_size)
                self.update_bounds(x2, y2, tile_size)

                patch1 = self.decode_nv12_to_bgr(raw1, tile_size)
                patch2 = self.decode_nv12_to_bgr(raw2, tile_size)

                if patch1 is not None: self.canvas[y1:y1+tile_size, x1:x1+tile_size] = patch1
                if patch2 is not None: self.canvas[y2:y2+tile_size, x2:x2+tile_size] = patch2
            except Exception as e:
                print(f"[解码错误]: {e}")
                continue

            packet_count += 1

            if packet_count % CHECK_INTERVAL == 0:
                results = self.model.predict(self.canvas, conf=CONF_THRESHOLD, verbose=False)[0]
                current_best = None
                y_conf = 0
                
                if len(results.boxes) > 0:
                    top_idx = results.boxes.conf.argmax()
                    y_conf = results.boxes.conf[top_idx].item()
                    current_best = {'box': results.boxes.xyxy[top_idx].cpu().numpy(), 'score': y_conf}

                stop_signal, iou = self.sccs_check(current_best)
                trans_ratio = packet_count / total_pkts if total_pkts > 0 else 0

                if self.csv_writer:
                    stable_cnt = self.history['stable_count'] if self.history else 0
                    self.csv_writer.writerow([packet_count, f"{trans_ratio:.4f}", f"{iou:.4f}", f"{y_conf:.4f}", f"{y_conf:.4f}", stable_cnt, timestamp_ms])
                    self.csv_file.flush()

                print(f"  [ID: {task_id}] 进度: {trans_ratio:.2%} | IoU: {iou:.4f} | 置信度: {y_conf:.4f} | 稳定计数: {stable_cnt}")

                # --- 根据开关决定是否保存中间过程 ---
                if SAVE_EVERY_CHECK:
                    self.save_result(packet_count, is_final=False)

                if stop_signal:
                    print(f"  >>> [锁定目标] 达到稳定阈值，继续检测。")
                    # 无论开关如何，最终锁定成功的图一定会保存一次
                    self.save_result(packet_count, is_final=True)
                    self.final_saved = True

if __name__ == "__main__":
    engine = ReceiverEngine()
    engine.run()
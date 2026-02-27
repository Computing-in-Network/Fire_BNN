import os
import socket
import struct
import time
from typing import Tuple
import numpy as np
import cv2
import csv

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# 简易 TCP 接收端，用于配合 TCP 发送端
# 要点：TCP 是字节流协议，需要手动按发送端固定包长 1088B 做边界重组

CONFIG = {
    "listen_ip": "127.0.0.1",
    "listen_port": 5005,
    # 接收后保存目录（每个包以二进制文件保存，便于离线分析）
    "out_dir": "./TCP/received",
    # 与发送端一致的固定包长度
    "pkt_len": 1088,
}

# YOLO / 推理相关配置（可按需调整或复制自 UDP/receive1.py ）
MODEL_PATH = "/home/yhli/yolo/yolo_projects/fire_mission/runs4/weights/best.pt"
CONF_THRESHOLD = 0.25
CHECK_INTERVAL = 25
STABLE_IOU_THR = 0.90
FINAL_CONF_THR = 0.70
STABLE_PATIENCE = 5
SAVE_EVERY_CHECK = True

# 引擎实例（稍后初始化）
ENGINE = None


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_header(header: bytes) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """
    解析 header 中的字段，返回常用字段方便打印。
    发送端使用的 pack 格式为: "<HBBHHHHHHHQQB"
    字段含义（按发送端顺序）:
      task_id(H), opcode(B), dtype(B), valid_len(H), tile_len(H), total_pkts(H),
      x1(H), y1(H), x2(H), y2(H), vaddr1(Q), timestamp_ms(Q), flag(B)
    """
    fmt = "<HBBHHHHHHHQQB"
    # 使用 unpack_from 可以在 header 缓冲的任意长度上解析
    vals = struct.unpack_from(fmt, header)
    (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
     x1, y1, x2, y2, vaddr1, timestamp_ms, flag) = vals

    return (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
            x1, y1, x2, y2, vaddr1, timestamp_ms, flag)


def handle_frame(frame: bytes, seq: int, out_dir: str, recv_time_ms: int):
    # frame 已保证为 CONFIG['pkt_len'] 长度
    header = frame[:64] #头部
    body = frame[64:]   #payload数据区域

    (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
     x1, y1, x2, y2, vaddr1, timestamp_ms, flag) = parse_header(header)

    # 数据区：先 tile1（384B），再 tile2（384B）
    tile1 = body[:384]
    tile2 = body[384:384*2]

    # 判定是否为单 tile（第二个 tile 为占位）
    single_tile = (x2 == 0xFFFF and y2 == 0xFFFF)

    fname = f"pkt_{seq:06d}_t{task_id}_x1{x1}_y1{y1}_x2{x2}_y2{y2}_ts{timestamp_ms}_f{flag}.bin"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "wb") as f:
        f.write(frame)

    # 诊断信息：接收时间、发送端时间戳差、是否单 tile、有效长度、首部摘要
    latency_ms = None
    try:
        latency_ms = int(recv_time_ms) - int(timestamp_ms)
    except Exception:
        latency_ms = None

    head_hex = header[:16].hex()
    print(f"[RX {seq}] bytes={len(frame)} task={task_id} total_pkts={total_pkts} flag={flag} single_tile={single_tile} valid_len={valid_len} tile_len={tile_len}")
    print(f"         coords: x1={x1},y1={y1} x2={x2},y2={y2} vaddr1={vaddr1}")
    print(f"         ts_sent={timestamp_ms} ts_recv={recv_time_ms} latency_ms={latency_ms} header16={head_hex} -> saved {fname}")

    # 如果推理引擎已初始化，则把该帧交给引擎进行画布更新与推理触发
    global ENGINE
    if ENGINE is not None:
        try:
            ENGINE.process_frame(frame, recv_time_ms, seq)
        except Exception as e:
            print(f"[!] 引擎处理帧出错: {e}")


class ReceiverEngine:
    def __init__(self):
        print(f"[Engine] 初始化推理引擎: model={MODEL_PATH}")
        self.model = None
        if YOLO is not None and MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"[Engine] 无法加载模型: {e}")
                self.model = None
        else:
            if YOLO is None:
                print("[Engine] ultralytics YOLO 未安装，推理被禁用")
            else:
                print("[Engine] 模型文件不存在，推理被禁用")

        self.canvas = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        self.history = None
        self.packet_count = 0
        self.csv_file = None
        self.csv_writer = None
        self.current_task_id = -1
        self.final_saved = False
        self.content_bounds = [float('inf'), float('inf'), 0, 0]

        os.makedirs('result', exist_ok=True)

    def reset(self, task_id):
        self.canvas.fill(128)
        self.history = None
        self.packet_count = 0
        self.final_saved = False
        self.current_task_id = task_id
        self.content_bounds = [float('inf'), float('inf'), 0, 0]

        csv_filename = os.path.join('result', f'task_{task_id}_metrics.csv')
        file_exists = os.path.isfile(csv_filename)
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
        self.csv_file = open(csv_filename, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["Sequence", "Trans_Ratio", "IoU", "Conf", "Smoothed", "Stable", "Time_ms"])

    def update_bounds(self, x, y, tile_size):
        self.content_bounds[0] = min(self.content_bounds[0], x)
        self.content_bounds[1] = min(self.content_bounds[1], y)
        self.content_bounds[2] = max(self.content_bounds[2], x + tile_size)
        self.content_bounds[3] = max(self.content_bounds[3], y + tile_size)

    def decode_nv12_to_bgr(self, data, tile):
        if not data:
            return None
        try:
            nv12_np = np.frombuffer(data, dtype=np.uint8).reshape((tile * 3 // 2, tile))
            return cv2.cvtColor(nv12_np, cv2.COLOR_YUV2BGR_NV12)
        except Exception:
            return None

    def sccs_check(self, current):
        if current is None:
            return False, 0
        if self.history is None:
            current['stable_count'] = 0
            self.history = current
            return False, 0

        iou_val = self._iou(current['box'], self.history['box'])
        stable_count = self.history.get('stable_count', 0) + 1 if iou_val > STABLE_IOU_THR else 0
        current['stable_count'] = stable_count
        self.history = current

        if current['score'] > FINAL_CONF_THR and stable_count >= STABLE_PATIENCE:
            return True, iou_val
        return False, iou_val

    def _iou(self, box1, box2):
        x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def save_result(self, seq, is_final=False):
        pad = 10
        x1 = max(0, int(self.content_bounds[0]) - pad)
        y1 = max(0, int(self.content_bounds[1]) - pad)
        x2 = min(self.canvas.shape[1], int(self.content_bounds[2]) + pad)
        y2 = min(self.canvas.shape[0], int(self.content_bounds[3]) + pad)

        if x1 >= x2 or y1 >= y2:
            return

        res_canvas = self.canvas[y1:y2, x1:x2].copy()
        if self.history:
            b = self.history['box'].astype(int)
            b_rel = [b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1]
            cv2.rectangle(res_canvas, (b_rel[0], b_rel[1]), (b_rel[2], b_rel[3]), (0, 0, 255), 2)
            cv2.putText(res_canvas, f"Fire: {self.history['score']:.2f}", (b_rel[0], b_rel[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        suffix = "_final" if is_final else ""
        save_name = os.path.join("result", f"task_{self.current_task_id}_seq_{seq}{suffix}.jpg")
        cv2.imwrite(save_name, res_canvas)

    def process_frame(self, frame: bytes, recv_time_ms: int, seq: int):
        # 解析 header
        header = frame[:64]
        (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
         x1, y1, x2, y2, vaddr1, timestamp_ms, flag) = parse_header(header)

        if task_id != self.current_task_id:
            self.reset(task_id)

        try:
            raw1 = frame[64:64 + tile_len]
            raw2 = frame[64 + tile_len:64 + 2 * tile_len]
            tile_size = 16
            self.update_bounds(x1, y1, tile_size)
            self.update_bounds(x2, y2, tile_size)

            patch1 = self.decode_nv12_to_bgr(raw1, tile_size)
            patch2 = self.decode_nv12_to_bgr(raw2, tile_size)

            if patch1 is not None:
                self.canvas[y1:y1+tile_size, x1:x1+tile_size] = patch1
            if patch2 is not None and not (x2 == 0xFFFF and y2 == 0xFFFF):
                self.canvas[y2:y2+tile_size, x2:x2+tile_size] = patch2
        except Exception as e:
            print(f"[Engine] 解码/写入画布错误: {e}")
            return

        self.packet_count += 1

        # 每 CHECK_INTERVAL 做一次推理和记录
        if self.packet_count % CHECK_INTERVAL == 0:
            if self.model is not None:
                try:
                    results = self.model.predict(self.canvas, conf=CONF_THRESHOLD, verbose=False)[0]
                except Exception as e:
                    print(f"[Engine] 推理错误: {e}")
                    results = None
            else:
                results = None

            current_best = None
            y_conf = 0
            if results is not None and len(results.boxes) > 0:
                top_idx = results.boxes.conf.argmax()
                y_conf = results.boxes.conf[top_idx].item()
                current_best = {'box': results.boxes.xyxy[top_idx].cpu().numpy(), 'score': y_conf}

            stop_signal, iou = self.sccs_check(current_best)
            trans_ratio = self.packet_count / total_pkts if total_pkts > 0 else 0

            if self.csv_writer:
                stable_cnt = self.history.get('stable_count', 0) if self.history else 0
                try:
                    self.csv_writer.writerow([self.packet_count, f"{trans_ratio:.4f}", f"{iou:.4f}", f"{y_conf:.4f}", f"{y_conf:.4f}", stable_cnt, timestamp_ms])
                    self.csv_file.flush()
                except Exception:
                    pass

            print(f"  [ID: {task_id}] 进度: {trans_ratio:.2%} | IoU: {iou:.4f} | 置信度: {y_conf:.4f} | 稳定计数: {stable_cnt}")

            if SAVE_EVERY_CHECK:
                self.save_result(self.packet_count, is_final=False)

            if stop_signal:
                print(f"  >>> [锁定目标] 达到稳定阈值，保存最终结果。")
                self.save_result(self.packet_count, is_final=True)
                self.final_saved = True


def start_server():
    listen = (CONFIG["listen_ip"], CONFIG["listen_port"])
    pkt_len = CONFIG["pkt_len"]
    out_dir = CONFIG["out_dir"]

    ensure_outdir(out_dir)

    # 初始化推理引擎
    global ENGINE
    ENGINE = ReceiverEngine() if 'ReceiverEngine' in globals() else None

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(listen)
    srv.listen(1)
    print(f"[*] TCP receiver listening on {listen}, saving to {out_dir}")

    while True:
        conn, addr = srv.accept()
        print(f"[+] Connection from {addr}")
        try:
            buffer = bytearray()
            seq = 0
            while True:
                data = conn.recv(4096)
                if not data:
                    print("[!] 连接已关闭 by peer")
                    break
                buffer.extend(data)

                # 尝试从 buffer 中按固定长度抽取完整帧
                while len(buffer) >= pkt_len:
                    frame = bytes(buffer[:pkt_len])
                    del buffer[:pkt_len]
                    seq += 1
                    recv_time_ms = int(time.time() * 1000)
                    handle_frame(frame, seq, out_dir, recv_time_ms)

                    # 可选：当接收到标记为最后包的帧时可选择停止当前连接
                    # 这里通过解析 header 中的 flag 字段做示例
                    # 解析 flag 很便捷：flag 在 header 最后一个字节，但我们已在 handle_frame 打印
                # end while
        except Exception as e:
            print(f"[!] 接收异常: {e}")
        finally:
            conn.close()
            print("[+] 连接关闭，等待下一个连接...")


if __name__ == "__main__":
    start_server()

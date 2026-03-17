import os
import socket
import struct
import time
import csv
import multiprocessing as mp

import numpy as np
import cv2
from ultralytics import YOLO


CONFIG = {
    "listen_ip": "192.168.96.3",
    "listen_port": 5005,
    "out_dir": "./received",
    "pkt_len": 1088,
    "idle_timeout_sec": 2.0,
    "queue_maxsize": 20000,
}

MODEL_PATH = "/workspace/bit-plane/fire.pt"
CONF_THRESHOLD = 0.25
CHECK_INTERVAL = 25
STABLE_IOU_THR = 0.90
FINAL_CONF_THR = 0.70
STABLE_PATIENCE = 5
SAVE_EVERY_CHECK = True

CANVAS_H = 3072
CANVAS_W = 3072


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_header(header: bytes):
    fmt = "<HBBHHHHHHHQQB"
    return struct.unpack_from(fmt, header)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


class WorkerEngine:
    def __init__(self):
        print(f"[Worker] 初始化推理引擎: model={MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print("[Worker] YOLO 模型加载成功")

        self.canvas = np.full((CANVAS_H, CANVAS_W, 3), 128, dtype=np.uint8)
        self.history = None
        self.packet_count = 0
        self.csv_file = None
        self.csv_writer = None
        self.current_task_id = -1
        self.final_saved = False
        self.content_bounds = [float("inf"), float("inf"), 0, 0]

        self.task_first_sent_ts = None
        self.expected_total_pkts = None
        self.last_recv_time_ms = None
        self.task_finished = False

        os.makedirs("result", exist_ok=True)

    def reset(self, task_id: int):
        self.canvas.fill(128)
        self.history = None
        self.packet_count = 0
        self.final_saved = False
        self.current_task_id = task_id
        self.content_bounds = [float("inf"), float("inf"), 0, 0]

        self.task_first_sent_ts = None
        self.expected_total_pkts = None
        self.last_recv_time_ms = None
        self.task_finished = False

        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass

        csv_filename = os.path.join("result", f"task_{task_id}_metrics.csv")
        self.csv_file = open(csv_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Detect_Time_ms",
            "Sequence",
            "Trans_Ratio",
            "YoloValue",
            "IoU",
            "Conf",
            "Smoothed",
            "SendFirst_to_Detect_ms",
            "Stable_Count",
            "Stable_Flag",
            "Stage"
        ])
        print(f"[Worker] 开始新任务 task_id={task_id}, csv={csv_filename}")

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
            return False, 0.0

        if self.history is None:
            current["stable_count"] = 0
            self.history = current
            return False, 0.0

        iou_val = calculate_iou(current["box"], self.history["box"])
        stable_count = self.history.get("stable_count", 0) + 1 if iou_val > STABLE_IOU_THR else 0
        current["stable_count"] = stable_count
        self.history = current

        if current["score"] > FINAL_CONF_THR and stable_count >= STABLE_PATIENCE:
            return True, iou_val
        return False, iou_val

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
            b = self.history["box"].astype(int)
            b_rel = [b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1]
            cv2.rectangle(res_canvas, (b_rel[0], b_rel[1]), (b_rel[2], b_rel[3]), (0, 0, 255), 2)
            cv2.putText(
                res_canvas,
                f"Fire: {self.history['score']:.2f}",
                (b_rel[0], max(15, b_rel[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        suffix = "_final" if is_final else ""
        save_name = os.path.join("result", f"task_{self.current_task_id}_seq_{seq}{suffix}.jpg")
        cv2.imwrite(save_name, res_canvas)

    def run_detection_and_record(self, total_pkts: int, stage: str):
        results = self.model.predict(self.canvas, conf=CONF_THRESHOLD, verbose=False)[0]

        current_best = None
        y_conf = 0.0

        if len(results.boxes) > 0:
            top_idx = results.boxes.conf.argmax()
            y_conf = float(results.boxes.conf[top_idx].item())
            current_best = {
                "box": results.boxes.xyxy[top_idx].cpu().numpy(),
                "score": y_conf
            }

        stop_signal, iou = self.sccs_check(current_best)
        trans_ratio = self.packet_count / total_pkts if total_pkts > 0 else 0.0

        detect_time_ms = int(time.time() * 1000)
        latency_since_first_send = ""
        if self.task_first_sent_ts is not None:
            try:
                latency_since_first_send = detect_time_ms - int(self.task_first_sent_ts)
            except Exception:
                latency_since_first_send = ""

        stable_cnt = self.history.get("stable_count", 0) if self.history else 0
        stable_flag = 1 if stop_signal else 0

        yolo_value = y_conf
        conf = y_conf
        smoothed = y_conf

        if self.csv_writer:
            self.csv_writer.writerow([
                detect_time_ms,
                self.packet_count,
                f"{trans_ratio:.4f}",
                f"{yolo_value:.4f}",
                f"{iou:.4f}",
                f"{conf:.4f}",
                f"{smoothed:.4f}",
                latency_since_first_send,
                stable_cnt,
                stable_flag,
                stage
            ])
            self.csv_file.flush()

        print(
            f"[Worker][ID:{self.current_task_id}] stage={stage} "
            f"progress={trans_ratio:.2%} | conf={y_conf:.4f} | IoU={iou:.4f} "
            f"| stable_count={stable_cnt} | delay={latency_since_first_send}ms"
        )

        if SAVE_EVERY_CHECK and stage == "periodic":
            self.save_result(self.packet_count, is_final=False)

        if stage.startswith("final"):
            self.save_result(self.packet_count, is_final=True)
            self.final_saved = True

    def finalize_task(self, reason: str):
        if self.task_finished or self.packet_count <= 0:
            return

        total_pkts = self.expected_total_pkts if self.expected_total_pkts is not None else self.packet_count
        stage_map = {
            "all_received": "final_all_received",
            "idle_timeout": "final_idle_timeout",
            "peer_closed": "final_peer_closed",
        }
        stage = stage_map.get(reason, "final_idle_timeout")

        print(f"[Worker] finalize_task: reason={reason}, packet_count={self.packet_count}, total_pkts={total_pkts}")
        self.run_detection_and_record(total_pkts, stage=stage)
        self.task_finished = True

    def process_frame(self, frame: bytes, recv_time_ms: int, seq: int):
        header = frame[:64]
        (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
         x1, y1, x2, y2, vaddr1, timestamp_ms, flag) = parse_header(header)

        if task_id != self.current_task_id:
            if self.packet_count > 0 and not self.task_finished:
                self.finalize_task("peer_closed")
            self.reset(task_id)

        if self.expected_total_pkts is None:
            self.expected_total_pkts = total_pkts

        self.last_recv_time_ms = recv_time_ms

        if self.task_first_sent_ts is None:
            self.task_first_sent_ts = timestamp_ms
        else:
            self.task_first_sent_ts = min(self.task_first_sent_ts, timestamp_ms)

        try:
            raw1 = frame[64:64 + tile_len]
            raw2 = frame[64 + tile_len:64 + 2 * tile_len]
            tile_size = 16

            patch1 = self.decode_nv12_to_bgr(raw1, tile_size)
            if patch1 is not None:
                self.canvas[y1:y1 + tile_size, x1:x1 + tile_size] = patch1
                self.update_bounds(x1, y1, tile_size)

            second_valid = not (x2 == 0xFFFF and y2 == 0xFFFF)
            if second_valid:
                patch2 = self.decode_nv12_to_bgr(raw2, tile_size)
                if patch2 is not None:
                    self.canvas[y2:y2 + tile_size, x2:x2 + tile_size] = patch2
                    self.update_bounds(x2, y2, tile_size)
        except Exception as e:
            print(f"[Worker] 解码/写入画布错误: {e}")
            return

        self.packet_count += 1
        self.task_finished = False

        if self.packet_count % CHECK_INTERVAL == 0:
            self.run_detection_and_record(total_pkts, stage="periodic")

        if self.expected_total_pkts is not None and self.packet_count >= self.expected_total_pkts:
            if not self.task_finished:
                print("[Worker] 已收满 total_pkts，执行最终检测")
                self.finalize_task("all_received")


def worker_loop(frame_queue: mp.Queue):
    engine = WorkerEngine()

    while True:
        item = frame_queue.get()
        if item is None:
            if engine.packet_count > 0 and not engine.task_finished:
                engine.finalize_task("peer_closed")
            break

        msg_type = item.get("type")

        if msg_type == "frame":
            engine.process_frame(
                frame=item["frame"],
                recv_time_ms=item["recv_time_ms"],
                seq=item["seq"]
            )
        elif msg_type == "eof":
            if engine.packet_count > 0 and not engine.task_finished:
                engine.finalize_task("peer_closed")
        elif msg_type == "idle_timeout":
            if engine.packet_count > 0 and not engine.task_finished:
                engine.finalize_task("idle_timeout")


def start_server():
    listen = (CONFIG["listen_ip"], CONFIG["listen_port"])
    pkt_len = CONFIG["pkt_len"]
    out_dir = CONFIG["out_dir"]
    idle_timeout_sec = CONFIG["idle_timeout_sec"]

    ensure_outdir(out_dir)

    frame_queue = mp.Queue(maxsize=CONFIG["queue_maxsize"])
    worker = mp.Process(target=worker_loop, args=(frame_queue,), daemon=True)
    worker.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(listen)
    srv.listen(1)

    print(f"[*] TCP receiver listening on {listen}, saving raw frames to {out_dir}")

    try:
        while True:
            conn, addr = srv.accept()
            conn.settimeout(idle_timeout_sec)
            print(f"[+] Connection from {addr}")

            try:
                buffer = bytearray()
                seq = 0

                while True:
                    try:
                        data = conn.recv(4096)
                    except socket.timeout:
                        print("[!] TCP 空闲超时")
                        frame_queue.put({"type": "idle_timeout"})
                        break

                    if not data:
                        print("[!] 连接已关闭 by peer")
                        frame_queue.put({"type": "eof"})
                        break

                    buffer.extend(data)

                    while len(buffer) >= pkt_len:
                        frame = bytes(buffer[:pkt_len])
                        del buffer[:pkt_len]
                        seq += 1
                        recv_time_ms = int(time.time() * 1000)

                        # 原始帧保存，保持你原来行为
                        header = frame[:64]
                        (task_id, opcode, dtype, valid_len, tile_len, total_pkts,
                         x1, y1, x2, y2, vaddr1, timestamp_ms, flag) = parse_header(header)
                        fname = f"pkt_{seq:06d}_t{task_id}_x1{x1}_y1{y1}_x2{x2}_y2{y2}_ts{timestamp_ms}_f{flag}.bin"
                        with open(os.path.join(out_dir, fname), "wb") as f:
                            f.write(frame)

                        frame_queue.put({
                            "type": "frame",
                            "frame": frame,
                            "recv_time_ms": recv_time_ms,
                            "seq": seq
                        })

            except Exception as e:
                print(f"[!] 接收异常: {e}")
            finally:
                conn.close()
                print("[+] 连接关闭，等待下一个连接...")

    except KeyboardInterrupt:
        print("\n[!] 收到退出信号，关闭中...")
    finally:
        try:
            frame_queue.put(None)
        except Exception:
            pass
        try:
            worker.join(timeout=3.0)
        except Exception:
            pass
        try:
            srv.close()
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start_server()
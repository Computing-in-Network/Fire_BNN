import os
import re
import socket
import struct
import subprocess
import time
from PIL import Image

# ==========================================================
# 配置中心 (针对 DPDK 链式 Docker 环境优化)
# ==========================================================
CONFIG = {
    # 修改为容器内的绝对路径 (假设您将 Fire_BNN_Tool 拷贝到了 /app/ 下)
    "bnn_script": "./batch_test.py",
    "result_file": "./result.txt",
    "patch_dir": "./test_images",
    
    # 核心修改：目标 IP 指向接收端容器 DPDK_1 的 IP [cite: 19, 23]
    "dst_ip": "192.168.10.237", 
    #"dst_ip": "127.0.0.1", 
    "dst_port": 5005,
    
    "sub_tile": 16,           # 转发窗口
    "send_interval_s": 0.0005   # 发送间隔
}

# 匹配文件名中的坐标 (逻辑不变)
XY_RE = re.compile(r"_x(\d+)_y(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
# 匹配 result.txt 中的 Score= 之后的分数 (逻辑不变)
SCORE_RE = re.compile(r"^([\w\.-]+)\s+.*?Score=([-\d\.eE]+)", re.IGNORECASE)

def get_nv12_bytes(img_patch: Image.Image) -> bytes:
    img = img_patch.convert("YCbCr")
    w, h = img.size
    y, cb, cr = img.split()
    y_bytes = y.tobytes()
    cb_sub = cb.resize((w // 2, h // 2), Image.BILINEAR)
    cr_sub = cr.resize((w // 2, h // 2), Image.BILINEAR)
    cb_bytes, cr_bytes = cb_sub.tobytes(), cr_sub.tobytes()
    uv = bytearray()
    for i in range(len(cb_bytes)):
        uv.append(cb_bytes[i]); uv.append(cr_bytes[i])
    return y_bytes + bytes(uv)

def build_dual_payload(block1, block2, task_id, total_pkts, is_last) -> bytes:
    """
    构造遵循 ALF 格式的 UDP Payload (保持 35 字节报头)
    """
    x1, y1, data1 = block1
    # x2, y2, data2 = block2
    
    timestamp_ms = int(time.time() * 1000)
    flag = 1 if is_last else 0

    if block2 is None:
        # 单 tile：第二个 tile 占位
        x2, y2 = 0xFFFF, 0xFFFF
        data2 = b"\x00" * 384
        valid_len = 384
    else:
        x2, y2, data2 = block2
        valid_len = 768
    
    # ALF 格式报头打包
    header = struct.pack(
        "<H B B H H H H H H H Q Q B",
        int(task_id),      # task_id
        1,                 # opcode
        1,                 # dtype
        int(valid_len),               # Valid_len
        384,               # Tile_len
        int(total_pkts),   # Total_pkts
        int(x1), int(y1),  # Tile1 坐标
        int(x2), int(y2),  # Tile2 坐标
        0,                 # Vaddr1
        timestamp_ms,      # Timestamp_ms
        flag               # 结束标志位
    )
    
    header = header.ljust(64, b'\x00')
    payload = header + data1 + data2
    return payload.ljust(1088, b'\x00')

def load_patches_in_order(patch_dir: str):
    """
    读取目录中所有 patch 文件，提取 (x,y,path)，按 (y,x) 排序（常用的扫描顺序）
    """
    items = []
    for fname in os.listdir(patch_dir):
        m = XY_RE.search(fname)
        if not m:
            continue
        x = int(m.group(1))
        y = int(m.group(2))
        items.append((y, x, os.path.join(patch_dir, fname)))

    # 顺序：先 y 再 x
    items.sort(key=lambda t: (t[0], t[1]))
    # 变成 (x,y,path)
    return [(x, y, path) for (y, x, path) in items]

def main():
    print("="*60)
    print(f"[*] 任务启动: DPDK 链式环境分发端 (目标: {CONFIG['dst_ip']})")
    print("="*60)

    # 使用 CONFIG 中定义的绝对路径
    #bnn_script = CONFIG["bnn_script"]
    result_file = CONFIG["result_file"]
    patch_dir = CONFIG["patch_dir"]

    # # --- 步骤 1: BNN 评分 ---
    # print(f"[*] 步骤 1: 正在运行 BNN 评分脚本 [{os.path.basename(bnn_script)}]...")
    # start_time = time.time()
    # try:
    #     subprocess.run(["python3", bnn_script], check=True)
    #     print(f"[+] 评分完成，耗时: {time.time() - start_time:.2f}s")
    # except Exception as e:
    #     print(f"[!] 错误: BNN 脚本运行失败: {e}")
    #     return

    # # --- 步骤 2: 解析分数 ---
    # scores = {}
    # if os.path.exists(result_file):
    #     with open(result_file, "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line: continue
    #             match = SCORE_RE.search(line)
    #             if match:
    #                 try:
    #                     scores[match.group(1)] = float(match.group(2))
    #                 except ValueError: continue
    #     print(f"[+] 成功加载了 {len(scores)} 条评分记录")

    # # --- 步骤 3: 扫描图片 ---
    # items = []
    # if os.path.exists(patch_dir):
    #     for fname in os.listdir(patch_dir):
    #         xy_match = XY_RE.search(fname)
    #         if not xy_match: continue
    #         score = scores.get(fname, 0.0)
    #         items.append((score, int(xy_match.group(1)), int(xy_match.group(2)), os.path.join(patch_dir, fname)))

    if not os.path.isdir(patch_dir):
        print(f"[!] 错误: {patch_dir}不存在。")
        return
    
    patches = load_patches_in_order(patch_dir)
    if not patches:
        print(f"[!] 错误: 在 {patch_dir} 未发现符合 *_x*_y*.jpg/png 的 16x16 小图")
        return

    # --- 步骤 4: 排序与预计算 ---
    total_pkts_per_task = (len(patches) + 1) // 2

    payloads = []
    pkt_idx = 0
    k = 0
    while k < len(patches):
        pkt_idx += 1
        is_last = (pkt_idx == total_pkts_per_task)

        x1, y1, p1 = patches[k]
        with Image.open(p1) as img1:
            img1 = img1.convert("RGB")
            data1 = get_nv12_bytes(img1)

        block1 = (x1, y1, data1)

        if k + 1 < len(patches):
            x2, y2, p2 = patches[k + 1]
            with Image.open(p2) as img2:
                img2 = img2.convert("RGB")
                data2 = get_nv12_bytes(img2)
            block2 = (x2, y2, data2)
            k += 2
        else:
            # 最后一张单独成包
            block2 = None
            k += 1

        payload = build_dual_payload(
            block1,
            block2,
            task_id=1,
            total_pkts=total_pkts_per_task,
            is_last=is_last
        )
        payloads.append(bytearray(payload))

    # --- 步骤 5: 建立 Socket 并通过 OVS/DPDK 链路发送 ---
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # dst = (CONFIG["dst_ip"], CONFIG["dst_port"])
    
    print(f"[*] 步骤 3: 开始发送 (总计包数: {total_pkts_per_task}) ...")
    print("-" * 60)

    # payloads = []
    # global_seq = 0
    # sub_blocks_pool = [] 

    # for idx, (score, ox, oy, fpath) in enumerate(items):
    #     img_name = os.path.basename(fpath)
    #     with Image.open(fpath) as img:
    #         img = img.convert("RGB")
    #         for i in range(4):
    #             for j in range(4):
    #                 sx, sy = ox + j*16, oy + i*16
    #                 patch = img.crop((j*16, i*16, j*16+16, i*16+16))
    #                 sub_blocks_pool.append((sx, sy, get_nv12_bytes(patch)))

    #                 if len(sub_blocks_pool) == 2:
    #                     global_seq += 1
    #                     is_last = (global_seq == total_pkts_per_task)
                        
    #                     payload = build_dual_payload(
    #                         sub_blocks_pool[0], 
    #                         sub_blocks_pool[1], 
    #                         task_id=1,
    #                         total_pkts=total_pkts_per_task,
    #                         is_last=is_last
    #                     )

    #                     payloads.append(bytearray(payload))
    #                     sub_blocks_pool = [] 

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (CONFIG["dst_ip"], CONFIG["dst_port"])

    start = time.perf_counter()
    N = len(payloads)
    pkt_bytes = len(payloads[0])
    bytes_sent = 0
    for i, payload in enumerate(payloads):
        #目标发送时刻
        target = start + i * CONFIG["send_interval_s"]
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        
        timestamp_ms = int(time.time() * 1000)
        struct.pack_into("<Q", payload, 26, timestamp_ms)

        sock.sendto(payload, dst)
        bytes_sent += len(payload)
                        
        if (i+1) % 8 == 0:
            print(f"    > 包已注入 DPDK 管道: {i+1}/{total_pkts_per_task}")
        if (i + 1) % 1000 == 0:
            t = time.perf_counter()
            elapsed = t - start
            pps = (i + 1) / elapsed
            mbps = (bytes_sent * 8) / elapsed / 1e6
            print(f"[sent={i+1}] elapsed={elapsed:.3f}s, pps={pps:.1f}, rate={mbps:.2f} Mbps")
    
    # sub_blocks_pool = []
    # if CONFIG["send_interval_s"] > 0:
    #     time.sleep(CONFIG["send_interval_s"])

    print("-" * 60)
    print(f"[*] 任务发送完成！数据已流向目标: {CONFIG['dst_ip']}")
    print("="*60)

if __name__ == "__main__":
    main()

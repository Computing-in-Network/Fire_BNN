import os
import re
import socket
import struct
import time
from PIL import Image

# ==========================================================
# 配置中心（TCP 发送端）
# ==========================================================
CONFIG = {
    "dst_ip": "127.0.0.1",   # 接收端容器 IP
    "dst_port": 5005,            # 接收端监听端口

    "patch_dir": "./TCP/test_images",  # 16x16 patch 目录（文件名含 _x*_y*）

    "send_interval_s": 0.0005,   # 发送间隔（秒）
}

# 匹配文件名中的坐标：xxx_x12_y34.jpg/png
XY_RE = re.compile(r"_x(\d+)_y(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

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


# ==========================================================
# 构造双 tile payload（1088B 定长）
# ==========================================================
def build_dual_payload(block1, block2, task_id, total_pkts, is_last) -> bytes:
  
    x1, y1, data1 = block1

    timestamp_ms = int(time.time() * 1000)
    flag = 1 if is_last else 0

    tile_len = 384

    if block2 is None:
        # 最后一包只有一个 tile：第二个 tile 用占位
        x2, y2 = 0xFFFF, 0xFFFF
        data2 = b"\x00" * tile_len
        valid_len = tile_len  # 表示只有 384B 有效
    else:
        x2, y2, data2 = block2
        valid_len = tile_len * 2  # 768B 有效

    header = struct.pack(
        "<HBBHHHHHHHQQB",
        int(task_id),     # task_id
        1,                # opcode
        1,                # dtype
        int(valid_len),   # valid_len
        int(tile_len),    # tile_len
        int(total_pkts),  # total_pkts
        int(x1), int(y1), # tile1 坐标
        int(x2), int(y2), # tile2 坐标
        0,                # vaddr1
        int(timestamp_ms),# timestamp_ms
        int(flag)         # flag (is_last)
    )

    # pad 到 64B 头
    header = header.ljust(64, b"\x00")

    payload = header + data1 + data2

    # 总长度 pad 到 1088
    return payload.ljust(1088, b"\x00")


# ==========================================================
# 扫描 patch 并按 y,x 排序
# ==========================================================
def load_patches_in_order(patch_dir: str):
    items = []
    for fname in os.listdir(patch_dir):
        m = XY_RE.search(fname)
        if not m:
            continue
        x = int(m.group(1))
        y = int(m.group(2))
        items.append((y, x, os.path.join(patch_dir, fname)))

    items.sort(key=lambda t: (t[0], t[1]))
    return [(x, y, path) for (y, x, path) in items]


# ==========================================================
# 主流程：TCP connect + sendall(1088B)
# ==========================================================
def main():
    patch_dir = CONFIG["patch_dir"]
    if not os.path.isdir(patch_dir):
        print(f"[!] 错误: patch_dir 不存在: {patch_dir}")
        return

    patches = load_patches_in_order(patch_dir)
    if not patches:
        print(f"[!] 错误: 在 {patch_dir} 未发现符合 *_x*_y*.jpg/png 的 patch")
        return

    total_pkts_per_task = (len(patches) + 1) // 2

    # 预构造所有 payload（可选：也可以边构造边发）
    # 注意：按照需求，先将所有包全部组好（与 UDP 端保持一致的组包格式），
    # 然后再按预设间隔统一发送。
    payloads = []
    pkt_idx = 0
    k = 0
    while k < len(patches):
        pkt_idx += 1
        is_last = (pkt_idx == total_pkts_per_task)

        x1, y1, p1 = patches[k]
        with Image.open(p1) as img1:
            data1 = get_nv12_bytes(img1.convert("RGB"))
        block1 = (x1, y1, data1)

        if k + 1 < len(patches):
            x2, y2, p2 = patches[k + 1]
            with Image.open(p2) as img2:
                data2 = get_nv12_bytes(img2.convert("RGB"))
            block2 = (x2, y2, data2)
            k += 2
        else:
            block2 = None
            k += 1

        payload = build_dual_payload(
            block1=block1,
            block2=block2,
            task_id=1,
            total_pkts=total_pkts_per_task,
            is_last=is_last
        )
        payloads.append(bytearray(payload))

    # ========= TCP 连接 =========
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TCP 与 UDP 的关键差异:
    # - 连接导向：TCP 需要先建立连接（connect），UDP 不需要（sendto 直接发）
    # - 传输语义：TCP 是字节流（stream），可能发生分段/合并；UDP 是报文边界保留的。
    # - 可靠性：TCP 提供重传与有序交付，UDP 不保证。
    # - 发送 API：TCP 使用 send()/sendall()；UDP 使用 sendto().
    # - 延迟调优：对小包高频发送，TCP 常设置 TCP_NODELAY 以禁用 Nagle 算法，减少延迟。
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    dst = (CONFIG["dst_ip"], CONFIG["dst_port"])
    print(f"[*] TCP connect -> {dst} ...")
    sock.connect(dst)
    print("[+] TCP connected.")

    print(f"[*] 开始发送，总记录数={len(payloads)}，每条=1088B")
    start = time.perf_counter()
    bytes_sent = 0

    for i, payload in enumerate(payloads):
        # 节拍控制
        target = start + i * CONFIG["send_interval_s"]
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)

        # 动态更新时间戳（保持与 UDP 端相同的偏移写法以便接收端解析一致）
        # UDP 版在 pack_into 的偏移为 26，这里保留相同偏移，确保 header 内 timestamp 字段一致。
        timestamp_ms = int(time.time() * 1000)
        struct.pack_into("<Q", payload, 26, timestamp_ms)

        # TCP 特殊点：使用 sendall 确保整个 1088B 数据写入到内核缓冲区。
        # 注意：因为 TCP 是流式协议，接收端需要按固定包长度（1088B）来做边界切分与重组。
        sock.sendall(payload)
        bytes_sent += len(payload)

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start
            pps = (i + 1) / elapsed
            mbps = (bytes_sent * 8) / elapsed / 1e6
            print(f"[sent={i+1}] elapsed={elapsed:.3f}s, pps={pps:.1f}, rate={mbps:.2f} Mbps")

    sock.close()
    print("[+] 发送完成，连接已关闭。")


if __name__ == "__main__":
    main()

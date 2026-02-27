import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_task_metrics(csv_file):
    # 确保保存路径也在 result 文件夹中
    output_dir = "result"
    
    print(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 清理表头空格
    df.columns = df.columns.str.strip()
    
    if 'Conf' not in df.columns:
        print(f"Error: Required column 'Conf' not found in {csv_file}")
        print(f"Available columns are: {list(df.columns)}")
        return

    # 提取任务 ID (处理路径中的 result/task_1_...)
    filename = os.path.basename(csv_file)
    try:
        task_id = filename.split('_')[1]
    except IndexError:
        task_id = "Unknown"
    
    # --- 图 1: Confidence vs. Transmission Ratio ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['Trans_Ratio'], df['Conf'], label='Confidence', color='steelblue', linewidth=2)
    plt.axhline(y=0.70, color='red', linestyle=':', alpha=0.6, label='Detection Threshold (0.70)')

    plt.title(f"Task {task_id}: Confidence vs. Transmission Ratio", fontsize=14)
    plt.xlabel("Transmission Ratio (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Confidence Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.ylim(0, 1.1)
    
    # 存回 result 文件夹
    save_path_conf = os.path.join(output_dir, f"task_{task_id}_confidence_plot.png")
    plt.savefig(save_path_conf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_conf}")

    # --- 图 2: IoU vs. Transmission Ratio ---
    plt.figure(figsize=(10, 6))
    if 'IoU' in df.columns:
        plt.plot(df['Trans_Ratio'], df['IoU'], label='Frame-to-Frame IoU', color='forestgreen', marker='o', markersize=4, linestyle='-')
        plt.axhline(y=0.90, color='orange', linestyle='--', label='Stability Threshold (0.90)')
    else:
        print(f"Warning: IoU column not found in {csv_file}")

    plt.title(f"Task {task_id}: IoU vs. Transmission Ratio", fontsize=14)
    plt.xlabel("Transmission Ratio (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Intersection over Union (IoU)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.ylim(0, 1.1)
    
    # 存回 result 文件夹
    save_path_iou = os.path.join(output_dir, f"task_{task_id}_iou_plot.png")
    plt.savefig(save_path_iou, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_iou}")

if __name__ == "__main__":
    # --- 修改：去 result 文件夹中搜索 CSV ---
    target_dir = "result"
    if not os.path.exists(target_dir):
        print(f"Directory '{target_dir}' does not exist.")
    else:
        csv_files = glob.glob(os.path.join(target_dir, "task_*_metrics.csv"))
        if not csv_files:
            print(f"No CSV files found in '{target_dir}' folder.")
        else:
            for f in csv_files:
                plot_task_metrics(f)
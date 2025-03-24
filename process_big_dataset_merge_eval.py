import os
import shutil
import json
from pathlib import Path
import glob

def merge_arrow_files():
    # 创建目标目录
    target_dir = "dataset/gift_eval_arrow_together/arrow_files"
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有arrow文件
    arrow_files = glob.glob("dataset/gift_eval_arrow/**/arrow_files/*.arrow", recursive=True)
    
    # 复制arrow文件到目标目录
    for i, src_file in enumerate(arrow_files):
        filename = f"chronos_{i}.arrow"
        dst_file = os.path.join(target_dir, filename)
        shutil.copy2(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")

    # 获取所有dataset_info文件并合并num_samples
    total_samples = 0
    dataset_info_files = glob.glob("dataset/gift_eval_arrow/**/dataset_files/dataset_info.json", recursive=True)
    
    for info_file in dataset_info_files:
        with open(info_file, 'r') as f:
            info = json.load(f)
            if 'num_sample' in info:
                total_samples += info['num_sample']
                print(f"Found {info['num_sample']} samples in {info_file}")
    
    target_dir = "dataset/gift_eval_arrow_together/dataset_files"
    os.makedirs(target_dir, exist_ok=True)
    # 创建新的dataset_info文件
    new_info = {
        "num_samples": total_samples
    }
    
    with open(os.path.join(target_dir, "dataset_info.json"), 'w') as f:
        json.dump(new_info, f, indent=2)
    
    print(f"\nTotal number of samples: {total_samples}")
    print(f"Merged dataset info saved to {target_dir}/dataset_info.json")

if __name__ == "__main__":
    merge_arrow_files()
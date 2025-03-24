import os
import shutil
import json
import pyarrow as pa
import pyarrow.ipc as ipc
import glob
from tqdm import tqdm

def process_arrow_files(source_dir,source_dir2, target_dir, batch_size=10000):
    """
    处理source_dir中的所有arrow文件，将符合条件的文件或合并后的文件保存到target_dir
    
    参数:
    - source_dir: 源目录，包含arrow文件的文件夹
    - target_dir: 目标目录，保存处理后的arrow文件
    - batch_size: 目标批次大小，默认10000
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有arrow文件
    arrow_files = glob.glob(f"{source_dir}/**/*.arrow", recursive=True)
    print(f"找到 {len(arrow_files)} 个arrow文件")
    arrow_files2 = glob.glob(f"{source_dir2}/**/*.arrow", recursive=True)
    print(f"找到 {len(arrow_files2)} 个arrow文件")
    arrow_files.extend(arrow_files2)
    print(f"找到 {len(arrow_files)} 个arrow文件")
    
    # 用于累计合并的记录批次
    accumulated_batches = []
    accumulated_count = 0
    
    # 用于统计信息
    total_samples = 0
    file_counts = {}  # 保存每个输出文件的样本数
    output_file_index = 0
    
    # 处理每个arrow文件
    for file_path in tqdm(arrow_files, desc="处理arrow文件"):
        try:
            # 读取arrow文件
            with pa.memory_map(file_path, 'r') as source:
                reader = ipc.RecordBatchFileReader(source)
                file_schema = reader.schema
                num_batches = reader.num_record_batches
                
                # 获取文件中的所有记录批次
                batches = [reader.get_record_batch(i) for i in range(num_batches)]
                
                # 计算样本数
                file_sample_count = sum(batch.num_rows for batch in batches)
                
                # 如果文件正好包含batch_size个样本，直接复制
                if file_sample_count == batch_size:
                    output_path = os.path.join(target_dir, f"batch_{output_file_index}.arrow")
                    shutil.copy2(file_path, output_path)
                    file_counts[output_path] = file_sample_count
                    total_samples += file_sample_count
                    output_file_index += 1
                    print(f"直接复制文件 {file_path} 到 {output_path}，样本数: {file_sample_count}")
                else:
                    # 将批次添加到累计列表
                    accumulated_batches.extend(batches)
                    accumulated_count += file_sample_count
                    
                    # 检查是否达到或超过目标大小
                    while accumulated_count >= batch_size:
                        # 提取batch_size个样本
                        output_batches = []
                        output_count = 0
                        remaining_batches = []
                        
                        for batch in accumulated_batches:
                            if output_count + batch.num_rows <= batch_size:
                                output_batches.append(batch)
                                output_count += batch.num_rows
                            else:
                                # 如果当前批次会导致超出batch_size，需要拆分
                                rows_needed = batch_size - output_count
                                if rows_needed > 0:
                                    # 拆分批次
                                    output_slice = batch.slice(0, rows_needed)
                                    remaining_slice = batch.slice(rows_needed)
                                    
                                    output_batches.append(output_slice)
                                    remaining_batches.append(remaining_slice)
                                    output_count += rows_needed
                                else:
                                    remaining_batches.append(batch)
                            
                            if output_count >= batch_size:
                                break
                        
                        # 将剩余的批次更新回accumulated_batches
                        for batch in accumulated_batches[len(output_batches):]:
                            if batch not in output_batches:
                                remaining_batches.append(batch)
                        
                        accumulated_batches = remaining_batches
                        accumulated_count -= output_count
                        
                        # 将合并的批次写入新的arrow文件
                        output_path = os.path.join(target_dir, f"batch_{output_file_index}.arrow")
                        
                        # 确保所有批次具有相同的架构
                        if output_batches:
                            with pa.OSFile(output_path, 'wb') as sink:
                                writer = ipc.RecordBatchFileWriter(sink, output_batches[0].schema)
                                for batch in output_batches:
                                    writer.write_batch(batch)
                                writer.close()
                            
                            file_counts[output_path] = output_count
                            total_samples += output_count
                            output_file_index += 1
                            print(f"写入合并文件 {output_path}，样本数: {output_count}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 处理剩余的数据
    if accumulated_batches:
        output_path = os.path.join(target_dir, f"batch_{output_file_index}.arrow")
        
        with pa.OSFile(output_path, 'wb') as sink:
            writer = ipc.RecordBatchFileWriter(sink, accumulated_batches[0].schema)
            for batch in accumulated_batches:
                writer.write_batch(batch)
            writer.close()
        
        file_counts[output_path] = accumulated_count
        total_samples += accumulated_count
        print(f"写入剩余数据到 {output_path}，样本数: {accumulated_count}")
    
    # 创建统计信息
    stats = {
        "total_samples": total_samples,
        "total_files": len(file_counts),
        "files": {os.path.basename(k): v for k, v in file_counts.items()}
    }
    
    # 将统计信息保存到JSON文件
    stats_path = os.path.join(target_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 创建dataset_info.json
    dataset_info = {
        "description": "Processed Arrow Dataset",
        "features": get_features_from_schema(file_schema) if 'file_schema' in locals() else {},
        "num_samples": total_samples
    }
    
    dataset_info_dir = os.path.join(target_dir, "dataset_files")
    os.makedirs(dataset_info_dir, exist_ok=True)
    
    with open(os.path.join(dataset_info_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n处理完成!")
    print(f"总样本数: {total_samples}")
    print(f"总文件数: {len(file_counts)}")
    print(f"统计信息已保存到 {stats_path}")
    print(f"数据集信息已保存到 {os.path.join(dataset_info_dir, 'dataset_info.json')}")
    
    return stats


def get_features_from_schema(schema):
    """从Arrow schema提取特征信息"""
    features = {}
    for field in schema:
        features[field.name] = {
            "dtype": str(field.type),
            "_type": "Value",
            "shape": [-1] if pa.types.is_list(field.type) else None
        }
    return features


if __name__ == "__main__":
    # 设置源目录和目标目录
    source_directory = "dataset/tempo"  # 源目录，包含多层arrow文件
    source_directory2 = "dataset/gift_eval_skip_48"  # 源目录，包含多层arrow文件
    target_directory = "dataset/gift_eval_skip_48_together"  # 目标目录，保存处理后的arrow文件
    
    # 处理arrow文件
    stats = process_arrow_files(source_directory, source_directory2, target_directory, batch_size=10000)
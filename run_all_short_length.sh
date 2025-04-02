#!/bin/bash

# 创建目录（如果不存在）
mkdir -p scripts/gift_single

# 定义起始端口
start_port=29720

# 定义所有数据集列表
all_datasets=(
    "hospital_M_short"
    "solar_W_short"
    "covid_deaths_D_short"
    "kdd_cup_2018_D_short" 
    "solar_D_short"
    "loop_seattle_D_short"
    "jena_weather_D_short"
    "ett2_D_short"
    "saugeen_M_short"
    "restaurant_D_short"
    "us_births_M_short"
    "m4_yearly_A_short"
    "ett2_W_short"
    "bitbrains_fast_storage_H_short"
    "electricity_D_short"
    "ett1_W_short"
    "car_parts_M_short"
    "hierarchical_sales_W_short"
    "m4_quarterly_Q_short"
    "electricity_W_short"
    "bitbrains_fast_storage_5T_long"
    "bitbrains_fast_storage_5T_medium"
    "bitbrains_fast_storage_5T_short"
    "bitbrains_rnd_5T_short"
    "bizitobs_service_10S_short"
    "electricity_15T_long"
    "electricity_15T_medium"
    "electricity_15T_short"
    "electricity_H_medium"
    "electricity_H_short"
    "jena_weather_10T_medium"
    "jena_weather_10T_short"
    "jena_weather_H_short"
    "loop_seattle_5T_long"
    "loop_seattle_5T_medium"
    "loop_seattle_5T_short"
    "loop_seattle_H_short"
    "solar_10T_medium"
    "solar_10T_short"
    "temperature_rain_D_short"
)

# 定义需要排除的数据集
exclude_datasets=(
   
)

# 计算差集 - 保留在all_datasets中但不在exclude_datasets中的元素
datasets=()
for dataset in "${all_datasets[@]}"; do
    skip=false
    for exclude in "${exclude_datasets[@]}"; do
        if [[ "$dataset" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == "false" ]]; then
        datasets+=("$dataset")
    fi
done

# 打印结果数据集数量
echo "Total datasets: ${#all_datasets[@]}"
echo "Excluded datasets: ${#exclude_datasets[@]}"
echo "Remaining datasets: ${#datasets[@]}"

# 读取原始脚本
template=$(cat scripts/gift_parallel_single_short_length.sh)

# 为每个数据集创建脚本
for dataset in "${datasets[@]}"; do
    # 替换数据集名称和端口号
    modified_script="${template//m_dense_D_short/$dataset}"
    modified_script="${modified_script//master_port=29529/master_port=$start_port}"
    
    # 保存到新文件
    echo "$modified_script" > "scripts/gift_single/${dataset}.sh"
    
    # 使新脚本可执行
    chmod +x "scripts/gift_single/${dataset}.sh"

    sbatch "scripts/gift_single/${dataset}.sh"
    
    # 递增端口号
    ((start_port++))
    
    echo "Created script for $dataset with port $((start_port-1))"
done

echo "All scripts created in gift_single/ directory"
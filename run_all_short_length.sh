#!/bin/bash

# 创建目录（如果不存在）
source .env
mkdir -p scripts/gift_single

# 定义起始端口
gpuid=1
# 定义起始端口
start_port=29720

# 定义所有数据集列表
all_datasets=(
    "electricity_15T_long"
    "jena_weather_10T_short"
    "loop_seattle_5T_medium"
    "solar_10T_medium"

    # "hospital_M_short"
    # "solar_W_short"
    # "covid_deaths_D_short"
    # "kdd_cup_2018_D_short" 
    # "solar_D_short"
    # "loop_seattle_D_short"
    # "jena_weather_D_short"
    # "ett2_D_short"
    # "saugeen_M_short"
    # "restaurant_D_short"
    # "us_births_M_short"
    # "m4_yearly_A_short"
    # "ett2_W_short"
    # "bitbrains_fast_storage_H_short"
    # "electricity_D_short"
    # "ett1_W_short"
    # "car_parts_M_short"
    # "hierarchical_sales_W_short"
    # "m4_quarterly_Q_short"
    # "electricity_W_short"
    # "bitbrains_fast_storage_5T_long"
    # "bitbrains_fast_storage_5T_medium"
    # "bitbrains_fast_storage_5T_short"
    # "bitbrains_rnd_5T_short"
    # "bizitobs_service_10S_short"
    
    # "electricity_15T_medium"
    # "electricity_15T_short"
    # "electricity_H_medium"
    # "electricity_H_short"
    # "jena_weather_10T_medium"
    
    # "jena_weather_H_short"
    # "loop_seattle_5T_long"
    
    # "loop_seattle_5T_short"
    # "loop_seattle_H_short"
    
    # "solar_10T_short"
    # "temperature_rain_D_short"
)

# 读取原始脚本
template=$(cat scripts/gift_parallel_single_short_length.sh)

# 为每个数据集创建脚本
for dataset in "${all_datasets[@]}"; do
    # 替换数据集名称和端口号
    modified_script="${template//m_dense_D_short/$dataset}"
    modified_script="${modified_script//master_port=29529/master_port=$start_port}"
    modified_script="${modified_script//CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=$gpuid}"
    
    # 保存到新文件
    echo "$modified_script" > "scripts/gift_single/${dataset}.sh"
    
    # 使新脚本可执行
    chmod +x "scripts/gift_single/${dataset}.sh"

    nohup bash "scripts/gift_single/${dataset}.sh" > "logs/0401/logging/${dataset}.log" 2>&1 &

    # 递增 GPU ID 并在达到 8 时重置为 0
    ((gpuid++))
    if [ $gpuid -eq 8 ]; then
        gpuid=0
    fi
    if [ $gpuid -eq 6 ]; then
        gpuid=7
    fi

    
    # 递增端口号
    ((start_port++))
    
    echo "Created script for $dataset with port $((start_port-1))"
done

echo "All scripts created in gift_single/ directory"
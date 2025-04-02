#!/bin/bash

# 创建目录（如果不存在）
mkdir -p scripts/gift_single_prob

# 定义起始端口
start_port=29520

# 定义所有数据集列表
all_datasets=(
    "electricity_H_short"
    "electricity_H_medium"
    "electricity_H_long"
    "hospital_M_short"
    "solar_W_short"
    "covid_deaths_D_short"
    "temperature_rain_D_short"
    "electricity_15T_short"
    "electricity_15T_medium"
    "electricity_15T_long"
    "bizitobs_l2c_5T_short"
    "bizitobs_l2c_5T_medium"
    "bizitobs_l2c_5T_long"
    "m4_weekly_W_short"
    "ett2_15T_short"
    "ett2_15T_medium"
    "ett2_15T_long"
    "bitbrains_rnd_H_short"
    "kdd_cup_2018_H_short"
    "kdd_cup_2018_H_medium"
    "kdd_cup_2018_H_long"
    "saugeen_D_short"
    "kdd_cup_2018_D_short"
    "m4_hourly_H_short"
    "solar_10T_short"
    "solar_10T_medium"
    "solar_10T_long"
    "us_births_W_short"
    "solar_D_short"
    "m_dense_D_short"
    "m4_monthly_M_short"
    "loop_seattle_D_short"
    "m_dense_H_short"
    "m_dense_H_medium"
    "m_dense_H_long"
    "ett2_D_short"
    "saugeen_W_short"
    "loop_seattle_H_short"
    "loop_seattle_H_medium"
    "loop_seattle_H_long"
    "solar_H_short"
    "solar_H_medium"
    "solar_H_long"
    "jena_weather_D_short"
    "restaurant_D_short"
    "us_births_M_short"
    "m4_yearly_A_short"
    "hierarchical_sales_D_short"
    "ett2_W_short"
    "ett2_H_short"
    "ett2_H_medium"
    "ett2_H_long"
    "bitbrains_fast_storage_5T_short"
    "bitbrains_fast_storage_5T_medium"
    "bitbrains_fast_storage_5T_long"
    "bitbrains_fast_storage_H_short"
    "electricity_D_short"
    "ett1_H_short"
    "ett1_H_medium"
    "ett1_H_long"
    "ett1_W_short"
    "m4_daily_D_short"
    "sz_taxi_H_short"
    "bitbrains_rnd_5T_short"
    "bitbrains_rnd_5T_medium"
    "bitbrains_rnd_5T_long"
    "sz_taxi_15T_short"
    "sz_taxi_15T_medium"
    "sz_taxi_15T_long"
    "bizitobs_l2c_H_short"
    "bizitobs_l2c_H_medium"
    "bizitobs_l2c_H_long"
    "car_parts_M_short"
    "ett1_15T_short"
    "ett1_15T_medium"
    "ett1_15T_long"
    "hierarchical_sales_W_short"
    "saugeen_M_short"
    "jena_weather_10T_short"
    "jena_weather_10T_medium"
    "jena_weather_10T_long"
    "us_births_D_short"
    "m4_quarterly_Q_short"
    "ett1_D_short"
    "bizitobs_application_10S_short"
    "bizitobs_application_10S_medium"
    "bizitobs_application_10S_long"
    "bizitobs_service_10S_short"
    "bizitobs_service_10S_medium"
    "bizitobs_service_10S_long"
    "loop_seattle_5T_short"
    "loop_seattle_5T_medium"
    "loop_seattle_5T_long"
    "jena_weather_H_short"
    "jena_weather_H_medium"
    "jena_weather_H_long"
    "electricity_W_short"
)

# 定义需要排除的数据集
exclude_datasets=(
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
template=$(cat scripts/gift_parallel_single_prob.sh)

# 为每个数据集创建脚本
for dataset in "${datasets[@]}"; do
    # 替换数据集名称和端口号
    modified_script="${template//m_dense_D_short/$dataset}"
    modified_script="${modified_script//master_port=29529/master_port=$start_port}"
    
    # 保存到新文件
    echo "$modified_script" > "scripts/gift_single_prob/${dataset}.sh"
    
    # 使新脚本可执行
    chmod +x "scripts/gift_single_prob/${dataset}.sh"

    sbatch "scripts/gift_single_prob/${dataset}.sh"
    
    # 递增端口号
    ((start_port++))
    
    echo "Created script for $dataset with port $((start_port-1))"
done

echo "All scripts created in gift_single/ directory"
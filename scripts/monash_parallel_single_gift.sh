#!/bin/bash
#SBATCH --job-name="m_dense_D_short"
#SBATCH --output="logs/m_dense_D_short.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcqc-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 12:00:00              # To enable the use of up to 8 GPUs          # To enable the use of up to 8 GPUs

# export CUDA_VISIBLE_DEVICES=0 #1 #4,5,6,7 #6,7 #4,5 #0,1,2,3 #6,7 #4,5 #0,1,2,3 #,4,5,6,7 #4,5,6,7 #0 #,1 0,1,2,3 #
source myenv/bin/activate

seq_len=336
model=TEMPO #TEMPO #PatchTST #_multi
electri_multiplier=3 # 3 times more data than the other small samples.
traffic_multiplier=3

#hospital_M_short: 长度(60)小于所需的序列长度(348)
#solar_W_short 长度(36)小于所需的序列长度(344)，将跳过
#covid_deaths_D_short 长度(152)小于所需的序列长度(366)，将跳过
# kdd_cup_2018_D_short 可能没有那么多sample
# solar_D_short 长度(275)小于所需的序列长度(366)，将跳过，变化的长度
# loop_seattle_D_short 长度(275)小于所需的序列长度(366)，将跳过
# jena_weather_D_short 长度(306)小于所需的序列长度(366)，将跳过
# ett2_D_short 不需要跳过，否则太短了。
# saugeen_M_short 不需要跳过，否则太短了。
# restaurant_D_short,长度(236)小于所需的序列长度(366)，将跳过
# us_births_M_short, 长度(216)小于所需的序列长度(348)，将跳过
# m4_yearly_A_short 长度(29)小于所需的序列长度(342)，将跳过
# ett2_W_short: 长度(79)小于所需的序列长度(344)，将跳过
# bitbrains_fast_storage_H_short: NAN ratio 0.9184 too high in item fastStorage_936, skipping
# electricity_D_short:  NAN ratio 0.9110070257611241 too high in item MT_133, skipping
# ett1_W_short: 长度(79)小于所需的序列长度(344)，将跳过
# car_parts_M_short: 长度(27)小于所需的序列长度(348)
# hierarchical_sales_W_short: 长度(27)小于所需的序列长度(348)，将跳过
# m4_quarterly_Q_short: target shape: (43,) 小于所需的序列长度(348)，将跳过
# electricity_W_short: 长度(184)小于所需的序列长度(344)，将跳过

for data_name in electricity_W_short #bitbrains_fast_storage_H_short #restaurant_D_short #jena_weather_D_short #solar_H_short #loop_seattle_H_short #saugeen_W_short #ett2_D_short #m_dense_D_short #kdd_cup_2018_H_long #bitbrains_rnd_H_short #ett2_15T_long # m4_weekly_W_short 
#bizitobs_l2c_5T_medium #bizitobs_l2c_5T_short
#hospital_M_short
#solar_W_short electricity_H_long #temperature_rain_D_short #hospital_M_short #jena_weather_H_short #electricity_H_short #jena_weather_H_short #electricity_H_short
do
for percent_mo in 100 #0.01 
do
for pred_len in  48 
do
for tmax in 20
do
for lr in 0.001 
do
for gpt_layer in 6 
do
for equal in 1
do
for prompt in 1 
do
mkdir -p logs/$model
# mkdir logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/
# mkdir logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer
# echo logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log

echo logs/$data_name'_'$seq_len.log

torchrun --nproc_per_node=2 --master_port=29529  train_TEMPO_parallel_single_data.py \
    --datasets $data_name \
    --eval_data $data_name \
    --target_data $data_name \
    --config_path ./configs/gift_eval.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./checkpoints/ \
    --model_id  $data_name'_'$seq_len \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 128 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 1 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --tmax $tmax \
    --cos 1 \
    --is_gpt 1 \
    --percent_mo $percent_mo #>> logs/$data_name'_'$seq_len.log


done
done
done
done
done
done
done
done

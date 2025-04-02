#!/bin/bash
#SBATCH --job-name="solar_H_long"
#SBATCH --output="logs/gift_dec_log_prob/solar_H_long.%j.%N.out"
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

source myenv/bin/activate

seq_len=336
model=TEMPO 
electri_multiplier=3 
traffic_multiplier=3

for data_name in solar_H_long 
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

echo logs/gift_dec_prob/$data_name'_'$seq_len.log

torchrun --nproc_per_node=2 --master_port=29646  train_TEMPO_parallel_single_data.py \
    --datasets $data_name \
    --eval_data $data_name \
    --target_data $data_name \
    --config_path ./configs/gift_eval.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./checkpoints_gift/ \
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
    --percent_mo $percent_mo >> logs/gift_dec/$data_name'_'$seq_len.log


done
done
done
done
done
done
done
done

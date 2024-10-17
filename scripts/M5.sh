#!/bin/bash
#SBATCH --job-name="Linear"
#SBATCH --output="./logs_mr/Linear.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdem-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00


source activate tempo
hostname


seq_len=170
model=PatchTST #DLinear #PatchTST #PatchTST #DLinear #NeuralCDE #GPT4TS #PatchTST #GPT4TS #PatchTST #DLinear #PatchTST #DLinear #PatchTST #DLinear #TEMPO #PatchTST 
electri_multiplier=1
traffic_multiplier=1


for percent in 100 
do
for pred_len in  28
do
for tmax in 20
do
for lr in 0.001 
do
for gpt_layer in 3 
do
for equal in 1 
do
for prompt in 1 
do
for datatype in FOODS HOBBIES HOUSEHOLD
do
for area in CA TX WI
do
mkdir -p logs/$model
# mkdir -p logs/$model/
# mkdir logs/$model/$datatype'_'$area.log
echo logs/$model/


python -u main_multi_6domain_release.py \
    --datasets M5_$area'_'$datatype \
    --target_data M5_$area'_'$datatype \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./lora_revin_6domain_checkpoints'_'$prompt/ \
    --model_id M5_mr_$model'_'$area'_'$datatype'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 10 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 256 \
    --learning_rate $lr \
    --train_epochs 100 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 1 \
    --c_out 1 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --tmax $tmax \
    --cos 1 \
    --is_gpt 1 \
    --loss_func prob > logs_mr/$model/$datatype'_'$area.log 2>&1


done
done
done
done
done
done
done
done
done

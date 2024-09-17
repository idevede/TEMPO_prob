#!/bin/bash
#SBATCH --job-name="198_prob"
#SBATCH --output="pretrain_198_prob.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcqc-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00


source activate tempo

seq_len=168
model=TEMPO #TEMPO #PatchTST 
electri_multiplier=1
traffic_multiplier=1


for percent in 100 
do
for pred_len in  30 #24 #96 192 336 720 
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
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/prob_TEMPO_no_pool_$model'_'$gpt_layer
echo logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/prob_TEMPO_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log


#electricity,traffic,weather
python main_multi_6domain_release.py \
    --datasets ETTh1,ETTm1,ETTh2,ETTm2,weather \
    --target_data ETTh1 \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./checkpoints/lora_revin_5domain_checkpoints'_'$prompt/ \
    --model_id prob_TEMPO'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 30 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 256 \
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
    --loss_func prob #>> logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/prob_TEMPO_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log


done
done
done
done
done
done
done

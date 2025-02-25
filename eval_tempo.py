# %% [markdown]
# # Quick Start: Running Foundation Model Moirai on gift-eval benchmark
# 
# This notebook shows how to run the Foundation Model Moirai on the gift-eval benchmark.
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment variable correctly before running this notebook.
# 
# We will use the `Dataset` class to load the data and run the model. If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook to learn more about the `Dataset` class. We are going to just run the model on two datasets for brevity. But feel free to run on any dataset by changing the `short_datasets` and `med_long_datasets` variables below.

# %%


# %%
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "m4_weekly"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "bizitobs_l2c/H"

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split()+ med_long_datasets.split()))
# all_datasets = list(set(med_long_datasets.split())) # short_datasets.split() +
dataset_properties_map = json.load(open("dataset_properties.json"))

# %%
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

# %%


# %%
from tempo.models.TEMPO import TEMPO
import torch
tempo_model = TEMPO.load_pretrained_model(
        device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu'),
        repo_id = "Melady/TEMPO",
        filename = "TEMPO-80M_v2.pth",
        cache_dir = "../TEMPO/checkpoints/TEMPO_checkpoints",
        # checkpoint_path='/home/defucao/workspace/TEMPO/checkpoints/Con2_Monash_TEMPO_6_prompt_learn_336_96_100/Con2_Monash_TEMPO_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth'
        checkpoint_path ='/home/defucao/workspace/TEMPO/checkpoints/Monash_1/Con1_Monash_TEMPO_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth'
)
     

import torch
import numpy as np
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from typing import Iterator, Dict, Any, List

class TEMPOPredictor(Predictor):
    def __init__(self, 
                 tempo_model,  # 你的 TEMPO PyTorch 模型
                 prediction_length: int,
                 freq: str,
                 num_samples: int = 100,  # 生成多少个采样路径
                 device=None):
        super().__init__(prediction_length)
        
        self.num_samples = num_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = tempo_model.to(device)
        self.prediction_length = prediction_length
        self.model.to(self.device)
        # self.model.eval()  # 设置为评估模式
    
    def predict(self, dataset, **kwargs) -> Iterator[SampleForecast]:
        """
        为数据集中的每个时间序列生成预测
        """
        for data_entry in dataset:
            # 提取必要数据
            target = data_entry.get('target')
            item_id = data_entry.get('item_id', None)
            start_date = data_entry.get('start', None)
            
            # 确保 target 是 numpy 数组
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            
            # 准备输入数据
            input_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)
            
            # 如果模型期望批量输入，调整形状
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)  # [batch_size=1, seq_len]
            
            # 生成多个样本预测
            samples_list = []
            
            with torch.no_grad():
                for _ in range(self.num_samples):
                    # 将你的模型调用封装在这里
                    # 假设模型返回的是下一个 prediction_length 个时间点的预测

                    # import pdb; pdb.set_trace()
                    prediction = self.model.predict(input_tensor, pred_length = self.prediction_length)
                    
                    # 如果预测包含在批量维度中，提取第一个元素
                    if len(prediction.shape) > 1 and prediction.shape[0] == 1:
                        prediction = prediction[0]
                    
                    # 转换为 numpy 并添加到样本列表
                    samples_list.append(prediction) #.cpu().numpy()
            
            # 合并所有样本为一个数组 [num_samples, prediction_length]
            samples = np.stack(samples_list) # 100, 48, 7 
            # print("The shape of final samples is: ", samples.shape)
            if len(samples.shape) ==3:
                samples = samples.transpose(0,2,1)
            # import pdb; pdb.set_trace()
            # 创建 GluonTS SampleForecast 对象
            forecast = SampleForecast(
                samples=samples,
                start_date=start_date,
                item_id=item_id
            )
            
            yield forecast

# %%
# from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# model = MoiraiForecast(
#     module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
#     prediction_length=1,
#     context_length=4000,
#     patch_size=32,
#     num_samples=100,
#     target_dim=1,
#     feat_dynamic_real_dim=0,
#     past_feat_dynamic_real_dim=0,
# )

# %% [markdown]
# ## Evaluation
# 
# Now that we have our predictor class, we can use it to predict on the gift-eval benchmark datasets. We will use the `evaluate_model` function to evaluate the model. This function is a helper function to evaluate the model on the test data and return the results in a dictionary. We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/moirai_small` folder.
# 
# The first column in the csv file is the dataset config name which is a combination of the dataset name, frequency and the term:
# 
# ```python
# f"{dataset_name}/{freq}/{term}"
# ```
# 

# %%
from gluonts.model import evaluate_model, evaluate_forecasts
import csv
import os
import time
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset

# Iterate over all available datasets

output_dir = "../results/tempo"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(
        [
            "dataset",
            "model",
            "eval_metrics/MSE[mean]",
            "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
        ]
    )

for ds_name in all_datasets:
    ds_key = ds_name.split("/")[0]
    print(f"Processing dataset: {ds_name}")
    terms = ["short", "medium", "long"]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and ds_name not in med_long_datasets.split():
            continue

        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"

        # Initialize the dataset, since Moirai support multivariate time series forecast, it does not require
        # to convert the original data into univariate
        # to_univariate = False if Dataset(name=ds_name, term=term,to_univariate=False).target_dim == 1 else True
        to_univariate = False
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        print(dataset.prediction_length)
        print(dataset.target_dim)
        # import pdb; pdb.set_trace()
        # set the Moirai hyperparameter according to each dataset, then create the predictor

        # model.hparams.prediction_length = dataset.prediction_length
        # model.hparams.target_dim = dataset.target_dim
        # model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim
        predictor = TEMPOPredictor(
            tempo_model=tempo_model,
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            num_samples=20  # 采样路径数量
        )
        # predictor = model.create_predictor(batch_size=512)

        season_length = get_seasonality(dataset.freq)

        forecasts = predictor.predict(dataset.test_data.input)

        # evaluate_forecasts_raw = evaluate_forecasts(
        # forecasts=forecasts,
        # test_data=dataset.test_data,
        # metrics=metrics,
        # axis=None,
        # batch_size=512,
        # mask_invalid_label=True,
        # allow_nan_forecast=False,
        # seasonality=season_length,
        # )

        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        # import pdb; pdb.set_trace()

        # Append the results to the CSV file
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    ds_config,
                    "TEMPO",
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                ]
            )

        print(f"Results for {ds_name} have been written to {csv_file_path}")

# %% [markdown]
# ## Results
# 
# Running the above cell will generate a csv file called `all_results.csv` under the `results/moirai_small` folder containing the results for the Moirai model on the gift-eval benchmark. The csv file will look like this:
# 

# %%
import pandas as pd

df = pd.read_csv("../results/tempo/all_results.csv")
df



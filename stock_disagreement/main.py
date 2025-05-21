import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from stock_disagreement import StockDisagreementTrainer
ROOT_PATH = "" # Your root path

def calculate_ic_rankic(res: pd.DataFrame, stock_labels: pd.DataFrame):
  
    label_cols = ["1_15_labelB", "5_15_labelB", "10_15_labelB"]
    sub_res = res[res["Date"] >= 20230102]
    df = sub_res[["Stock", "Date", "Signal", "Signal_mean", "Signal_std"]].merge(stock_labels[["Stock", "Date"] + label_cols], on=["Stock", "Date"])
    for label_col in label_cols:
        temp_df = df[[label_col, "Signal", "Signal_mean", "Signal_std"] + ["Stock", "Date"]].copy()
      
        rankic_values = temp_df.groupby(by = ["Date"])[["Stock",label_col, "Signal"]].apply(
            lambda x: np.corrcoef(x[label_col].rank(), x["Signal"].rank())[0, 1]
        )
        ic_values = temp_df.groupby(by = ["Date"])[["Stock",label_col, "Signal"]].apply(
            lambda x: np.corrcoef(x[label_col], x["Signal"])[0, 1]
        )
        rankic_values_mean = temp_df.groupby(by = ["Date"])[["Stock",label_col, "Signal_mean"]].apply(
            lambda x: np.corrcoef(x[label_col].rank(), x["Signal_mean"].rank())[0, 1]
        )
        ic_values_mean = temp_df.groupby(by = ["Date"])[["Stock",label_col, "Signal_mean"]].apply(
            lambda x: np.corrcoef(x[label_col], x["Signal_mean"])[0, 1]
        )
        rankic_values_std = temp_df.groupby(by = ["Date"])[["Stock",label_col, "Signal_std"]].apply(
            lambda x: np.corrcoef(x[label_col].rank(), x["Signal_std"].rank())[0, 1]
        )
        ic_values_std = temp_df.groupby(by = ["Date"])[["Stock", label_col, "Signal_std"]].apply(
            lambda x: np.corrcoef(x[label_col], x["Signal_std"])[0, 1]
        )

        rankic_ir = rankic_values.mean() / rankic_values.std()
        rankic = rankic_values.mean()
        ic = ic_values.mean()
        icir = ic_values.mean() / ic_values.std()
        print(f"for {label_col}, ic: {ic}, icir: {icir}, rankic: {rankic}, rankicir: {rankic_ir}")
        print(f"for {label_col}, ic_mean: {ic_values_mean.mean()}, icir_mean: {ic_values_mean.mean() / ic_values_mean.std()}, rankic_mean: {rankic_values_mean.mean()}, rankicir_mean: {rankic_values_mean.mean() / rankic_values_mean.std()}")
        print(f"for {label_col}, ic_std: {ic_values_std.mean()}, icir_mean: {ic_values_std.mean() / ic_values_std.std()}, rankic_mean: {rankic_values_std.mean()}, rankicir_mean: {rankic_values_std.mean() / rankic_values_std.std()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stock disagreement argument parser.")
    parser.add_argument("--num_investor_type", type=int, default= 16)
    parser.add_argument("--num_agents_per_investor", type=int, default= 32)
    parser.add_argument("--stock_pool", type=str, default="ih", choices=["ih", "csi_300", "csi_500", "csi_1000", "start_up_100"])
    parser.add_argument("--stock_num", type=int, default=20)
    parser.add_argument("--selected_stock_num", type=int, default=5)
    parser.add_argument("--start_date", type=int, default= 20221202)
    parser.add_argument("--end_date", type=int, default= 20240102)
    parser.add_argument("--use_prev_stock", type=bool, default=True)
    parser.add_argument("--use_self_reflection", type=bool, default=False)
    parser.add_argument("--use_macro_data", type=bool, default=True)
    parser.add_argument("--use_agent_distribution_modification", type=bool, default=True)
    parser.add_argument("--optimizer_look_back_window", type=int, default=5)
    parser.add_argument("--allow_possible_data_leakage", type=bool, default=False)

    args = parser.parse_args()

    stock_pool_name = args.stock_pool
    assert isinstance(stock_pool_name, str)
    stock_pool = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/{stock_pool_name}.parq")
    industry = pd.read_parquet("{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/stock_basic_data.parq")
    stock_pool = stock_pool.merge(industry[["Stock", "Name", "Industry"]], on=["Stock"], how="left")
    # Change to your label file url here.
    stock_labels = pd.read_parquet("{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/all_ashare_label.parq")

    trainer = StockDisagreementTrainer(
        num_investor_type=args.num_investor_type,
        num_agents_per_investor=args.num_agents_per_investor,
        stock_selector_for_per_investor=args.selected_stock_num,
        stock_pool=stock_pool,
        stock_labels=stock_labels,
        stock_num = args.stock_num,
        start_date = args.start_date,
        end_date = args.end_date,
        use_prev_stock = args.use_prev_stock,
        use_self_reflection = args.use_self_reflection,
        use_macro_data = args.use_macro_data,
        use_agent_distribution_modification = args.use_agent_distribution_modification,
        look_back_window = 10,
        optimizer_look_back_window = args.optimizer_look_back_window,
        data_leakage = args.allow_possible_data_leakage
    )
    res = trainer.run()
    res.to_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/res/{stock_pool_name}_{args.num_agents_per_investor}_{args.num_investor_type}_{args.use_macro_data}_{args.use_agent_distribution_modification}_{args.optimizer_look_back_window}_{args.allow_possible_data_leakage}_{args.start_date}_{args.end_date}_{args.use_self_reflection}_std.parq")
    with open(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/res/dist_{stock_pool_name}_{args.num_agents_per_investor}_{args.num_investor_type}_{args.use_macro_data}_{args.use_agent_distribution_modification}_{args.optimizer_look_back_window}_{args.allow_possible_data_leakage}_{args.start_date}_{args.end_date}_{args.use_self_reflection}_{args.stock_num}_positive.pkl", "wb") as f:
        pkl.dump(trainer.agent_distributions, f)
    calculate_ic_rankic(res, stock_labels)

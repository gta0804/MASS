from stock_disagreement import DTML
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Sampler
import os

class StockDataset(Dataset):

    def __init__(
        self,
        stock_features_data_path: str,
        stock_label_data_path: str,
        stock_news_data_path: str | None = None,
        use_tick_data: bool = False,
        T: int = 20,
        start_date: int = 20180101,
        end_date: int = 20211231,
    ) -> None:
        """Stock Data Class.

        Args:
            data_path (str): file path of stock features in parquet.
            stock_label_data_path(str): file path of stock labels in parquet.
            stock_news_data_path(str | None):
            T(int): look back length.
            start_date(int): data start date.
            end_date(int): data end date.
        """
        self.data = pd.read_parquet(
            stock_features_data_path
        )
        feature_columns = np.setdiff1d(self.data.keys(), ["Stock", "Date", "Time"])
        for column in feature_columns:
            self.data[column] = self.data.groupby("Date")[column].transform(lambda x : (x - x.mean()) / x.std())
        self.label = pd.read_parquet(
            stock_label_data_path
        )
        self.start_date = start_date
        self.end_date = end_date
        self.use_tick_data = use_tick_data
        self.T = T
        self.process_data()
        self.samples: list[int] = self.get_sample_index()
        self.dates = self.data.Date.values[self.samples]
        self.stocks = self.data.Stock.values[self.samples]
        label_columns = ["10_15_labelB"]

        self.features = self.data[feature_columns].values
        self.labels = self.data[label_columns].values

    def process_data(self) -> None:

        self.data = self.data[self.data.Date >= self.start_date]
        self.data = self.data[self.data.Date <= self.end_date]
        self.data.sort_values(["Stock", "Date"], inplace=True)
        self.label = self.label[self.label.Date >= self.start_date]
        self.label = self.label[self.label.Date <= self.end_date]

        # if self.use_tick_data:
        #     self.data["Time"] = (self.data["Time"].astype(int)).astype(str).str.zfill(4)
        #     self.data["Datetime"] = pd.to_datetime(
        #         self.data["Date"]
        #         + " "
        #         + self.data["Time"].str[:-2]
        #         + ":"
        #         + self.data["Time"].str[-2:]
        #     )
        # else:
        #     self.data["Date"] = pd.to_datetime(self.data["Date"])

        self.label["Time"] = self.label["Time"].astype(int)

        # if not self.use_tick_data:
        #     self.label = self.label[self.label["Time"] == 1445]
        self.data = self.data.merge(self.label, on=["Stock", "Date"], how="inner")
        # self.data["Date"] = self.data["Date"].astype(str)
        # self.data = self.data.dropna()
        self.data.reset_index(drop=True, inplace=True)

    def get_sample_index(self) -> list[int]:
        """Get sample indexs."""
        samples: list[int] = []
        stocks, ids, counts = np.unique(
            self.data["Stock"], return_index=True, return_counts=True
        )
        print(f"Contains {len(ids)} stocks...")
        for i in range(len(ids)):
            # print(
            #     "Stocks: "
            #     + str(stocks[i])
            #     + " Start date: "
            #     + str(self.data.iloc[ids[i]]["Date"])
            #     + " End date: " + str(self.data.iloc[ids[i] + counts[i] - 1]["Date"]),
            # )
            st_index = ids[i] + self.T - 1
            ed_index = ids[i] + counts[i]
            samples.extend(list(range(st_index, ed_index)))
        return samples

    def __getitem__(self, index: int) -> dict:
        id = self.samples[index]
        item_dict = {}
        item_dict["date"] = self.dates[index]
        item_dict["stock"] = self.stocks[index]
        item_dict["feature"] = self.features[(id - self.T + 1) : (id + 1)]
        item_dict["label"] = self.labels[id]
        return item_dict

    def __len__(self) -> int:
        return len(self.samples)


class StockDailySampler(Sampler):
    def __init__(self, dataset: StockDataset, shuffle: bool = True):
        self.sel_dates = np.unique(dataset.dates)
        self.all_dates = dataset.dates
        self.sample_indices = list(range(len(self.sel_dates)))
        if shuffle:
            np.random.shuffle(self.sample_indices)
        self.batchs = []
        for index in self.sample_indices:
            date = self.sel_dates[index]
            batch = np.where((self.all_dates == date))[0].tolist()
            self.batchs.append(batch)

    def __iter__(self):
        for batch in self.batchs:
            yield batch

    def __len__(self):
        return len(self.sample_indices)
    

def get_dataloaders(pool_name:str):
    dl_train = StockDataset(stock_features_data_path="/users/hy/stock_prediction/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq",
                            stock_label_data_path="/users/hy/stock_prediction/stock_prediction_benchmark/stock_disagreement/dataset/all_ashare_label.parq",
                            start_date=20210101,
                            end_date=20230101,
                            T=20)
    dl_test = StockDataset(stock_features_data_path="/users/hy/stock_prediction/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq",
                            stock_label_data_path="/users/hy/stock_prediction/stock_prediction_benchmark/stock_disagreement/dataset/all_ashare_label.parq",
                            start_date=20230101,
                            end_date=20240101,
                            T=20)
    
    return dl_train, dl_test


def train(config_path: pathlib.Path):
    """Train model."""
    dl_train, dl_val, dl_test, train_info = parse_train_config(config_path)
    # Set seed
    set_seed(seed=train_info["seed"])
    gpu_id = train_info["gpu_id"]
    device = torch.device(f"cuda:{gpu_id}")
    # Get dataloaders
    # Get models
    model = get_model(train_info["model"], device=device)
    # Get loss_fun
    loss_fun = get_loss_fun(train_info["loss"])
    # Get optimizer
    optimizer = get_optimizer(model, train_info["optimizer"])
    save_dir = train_info["save_dir"]
    dl_train = init_data_loader(dl_train, shuffle=True, drop_last=True)
    dl_val = init_data_loader(dl_val, shuffle=False, drop_last=True)
    dl_test = init_data_loader(dl_test, shuffle=True, drop_last=True)
    # Train model
    best_epoch = train_model(
        num_epochs=train_info["train_epochs"],
        model=model,
        loss_fun=loss_fun,
        optimizer=optimizer,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        device=device,
        save_dir=save_dir,
    )
    model.load_state_dict(torch.load(save_dir + f"epoch_{best_epoch}.pth"))
    # Test model
    metrics = val_on_epoch(model=model, dl=dl_test, device=device)
    res = [f"{k}: {v};" for k, v in metrics.items()]
    print("Test performance: " + " ".join(res))


def test_pred(exp_path: pathlib.Path, epoch_num: int):
    """Test inference for model."""
    exp_path = pathlib.Path(exp_path)
    dataset_info, train_info, save_dir = parse_train_config(
        exp_path / "config.json", test=True
    )
    gpu_id = train_info["gpu_id"]
    device = torch.device(f"cuda:{gpu_id}")
    # Get dataloaders
    dl_train, dl_val, dl_test = get_dataloaders(dataset_info)
    # Get models
    model = get_model(train_info["model"], device=device)
    model.load_state_dict(torch.load(save_dir / "models" / f"epoch_{epoch_num}.pth"))
    # Test model
    _, metrics = val_on_epoch(model=model, dl=dl_test, device=device)
    res = [f"{k}: {v};" for k, v in metrics.items()]
    print("Test performance: " + " ".join(res))
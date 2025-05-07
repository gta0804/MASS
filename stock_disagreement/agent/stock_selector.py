import pandas as pd
import numpy as np
from typing import Iterator, Any, Optional
import random


class BasicStockSelector():
    def __init__(self, stock_num: int, 
                 stock_pool: pd.DataFrame, 
                 start_date: int | None = None,
                 end_date: int | None = None):
        self.stock_num = stock_num
        self.stock_pool = stock_pool
        self.stock_pool = self.stock_pool.sort_values(["Stock", "Date"])
        if start_date is None:
            self.start_date = self.stock_pool["Date"].min()
        else:
            self.start_date = start_date
        if end_date is None:
            self.end_date = self.stock_pool["Date"].max()
        else:
            self.end_date = end_date
        self.stock_pool = self.stock_pool.query("Date >= @self.start_date and Date <= @self.end_date")
        self.stock_iterator = self._get_cross_section()
        self.current_date = None
        self.current_data = None
    
    def _get_cross_section(self) -> Iterator[tuple[int, pd.DataFrame]]:
        for (date, group) in self.stock_pool.groupby("Date"):
            yield (date, group)

    def get_next(self) -> bool:
        try:
            self.current_date, self.current_data = next(self.stock_iterator)
            return True
        except StopIteration:
            return False
    
    def select_stock_for_llm(self) -> list[str]:
        pass

class RandomStockSelector(BasicStockSelector):
    def __init__(self, stock_num: int,
                 stock_pool: pd.DataFrame, 
                 start_date: int | None = None, 
                 end_date: int | None = None):
        super().__init__(stock_num, stock_pool, start_date, end_date)
    
    def select_stock_for_llm(self, date:int = None) -> list[str]:
        if date is None:
            return random.choices(self.current_data["Stock"].to_list(),k=self.stock_num)
        else:
            current_data = self.stock_pool[self.stock_pool["Date"] == date].copy()
            return random.choices(current_data["Stock"].to_list(),k=self.stock_num)


class IndustryEqualStockSelector(BasicStockSelector):
    def __init__(self, stock_num, stock_pool, start_date=None, end_date=None):
        super().__init__(stock_num, stock_pool, start_date, end_date)

    def select_stock_for_llm(self, date:int=None) -> list[str]:
        if date is None:
            assert "Industry" in self.current_data.columns
            self.current_data["weight"] = 1
            weights = self.current_data.groupby("Industry")["weight"].transform(lambda x: x / x.sum())
            return random.choices(self.current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())
        else:
            current_data = self.stock_pool[self.stock_pool["Date"] == date].copy()
            assert "Industry" in current_data.columns
            current_data["weight"] = 1
            weights = current_data.groupby("Industry")["weight"].transform(lambda x: x / x.sum())
            return random.choices(current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())


class IndustryBasisStockSelector(BasicStockSelector):
    def __init__(self, stock_num, stock_pool, start_date = None, end_date = None, basis_industry = None):
        super().__init__(stock_num, stock_pool, start_date, end_date)
        self.basis_industry = basis_industry
    
    def select_stock_for_llm(self, basis: float = 0.2, date: int = None) -> list[str]:
        if date is None:
            assert "Industry" in self.current_data.columns
            self.current_data["weight"] = np.where(self.current_data["Industry"].isin(self.basis_industry), 1, basis)
            weights = self.current_data.groupby("Industry")["weight"].transform(lambda x: x / x.sum())
            return random.choices(self.current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())
        else:
            current_data = self.stock_pool[self.stock_pool["Date"] == date].copy()
            assert "Industry" in current_data.columns
            current_data["weight"] = np.where(current_data["Industry"].isin(self.basis_industry), 1, basis)
            weights = current_data.groupby("Industry")["weight"].transform(lambda x: x / x.sum())
            return random.choices(current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())        


class MVEqualStockSelector(BasicStockSelector):
    def __init__(self, stock_num, stock_pool, start_date = None, end_date = None):
        super().__init__(stock_num, stock_pool, start_date, end_date)
    
    def select_stock_for_llm(self, num_groups: int = 5, date: int = None) -> list[str]:
        if date is None:
            assert "FREE_MV" in self.current_data.columns
            self.current_data["rank"] = self.current_data.groupby("Date")["FREE_MV"].rank(ascending=False)
            self.current_data["count"] = self.current_data.groupby("Date")["Stock"].transform("count")
            self.current_data["group"] = ((self.current_data["rank"] - 1) // (self.current_data["count"] / num_groups)).astype(int)
            self.current_data["weight"] = 1
            weights = self.current_data.groupby("group")["weight"].transform(lambda x: x / x.sum())
            return random.choices(self.current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())
        else:
            current_data = self.stock_pool[self.stock_pool["Date"] == date].copy()
            assert "FREE_MV" in current_data.columns
            current_data["rank"] = current_data.groupby("Date")["FREE_MV"].rank(ascending=False)
            current_data["count"] = current_data.groupby("Date")["Stock"].transform("count")
            current_data["group"] = ((current_data["rank"] - 1) // (current_data["count"] / num_groups)).astype(int)
            current_data["weight"] = 1
            weights = current_data.groupby("group")["weight"].transform(lambda x: x / x.sum())
            return random.choices(current_data["Stock"].to_list(), k=self.stock_num, weights=weights.to_list())

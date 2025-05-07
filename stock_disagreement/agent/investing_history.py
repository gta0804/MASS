from collections import deque
from typing import Tuple, Optional

class InvestingHistory():
    """记录每个Agent的投资决策历史."""
    def __init__(self) -> None:
        self.records = deque()
        self.history_stock_list = deque()
        self.chosen_stock_list = deque()

    def add_records(self, record: str, history_stocks: list[str], chosen_stocks: list[str]) -> None:
        self.records.append(record)
        self.history_stock_list.append(history_stocks)
        self.chosen_stock_list.append(chosen_stocks)

    def get_record(self, n_days_ago: int = 1) -> Tuple[bool, Optional[str]]:
        if n_days_ago <= 0:
            raise ValueError(f"{n_days_ago} is illegal, please input positive integers.")
        total_records = len(self.records)
        if n_days_ago > total_records:
            return False, {}
        return True, self.records[-n_days_ago]
 
    def get_history_stocks(self, n_days_ago: int = 1) -> Tuple[bool, Optional[list[str]]]:
        if n_days_ago <= 0:
            raise ValueError(f"{n_days_ago} is illegal, please input positive integers.")
        total_stock_records = len(self.history_stock_list)
        if n_days_ago > total_stock_records:
            return False, []
   
        return True, self.history_stock_list[-n_days_ago]
    
    def get_chosen_stocks(self, n_days_ago: int = 1) -> Tuple[bool, Optional[list[str]]]:
        if n_days_ago <= 0:
            raise ValueError(f"{n_days_ago} is illegal, please input positive integers.")
        total_stock_records = len(self.chosen_stock_list)
        if n_days_ago > total_stock_records:
            return False, []
    
        return True, self.chosen_stock_list[-n_days_ago]

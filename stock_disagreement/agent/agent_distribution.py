import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from stock_disagreement.agent import InvestmentAnalyzer
import bisect
import concurrent
import random
from tqdm import tqdm
import time

class BaseOptimizer(ABC):
    """
    基类 提供Agent 分布优化的初始逻辑.
    """
    interval: int = 4
    def __init__(self,
                max_iter:int = 20,
                look_back_window: int = 5):
        super().__init__()
        self.max_iter = max_iter
        self.look_back_window = look_back_window
    

    def calcluate_fitness(self, investment_res: pd.DataFrame, stock_labels: pd.DataFrame, alpha: float = 0.5) -> float:
        sub_res = investment_res.copy()
        sub_res = investment_res.merge(stock_labels, on = ["Stock", "Date"], how="inner")
        label_col = "5_15_labelB"
        signal_col = "Signal_std"
        rankic_values = sub_res.groupby(by = ["Date"])[["Stock",label_col, signal_col]].apply(
            lambda x: np.corrcoef(x[label_col].rank(), x[signal_col].rank())[0, 1]
        )

        return rankic_values.mean()
    
    @abstractmethod
    def optimize(self,
                investment_res: InvestmentAnalyzer,
                dates: list[int],
                current_date: int,
                stock_pool: pd.DataFrame,
                distribution: list[float]) -> list[float]:
        pass



class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    模拟退火算法.
    """
    mean: float = 0
    std_dev: float = 0.25
    def __init__(self,
                 look_back_window: int = 5,
                 max_iter:int = 20,
                 init_temp: float = 0.5,  
                 cooling_rate: float = 0.95,
                 ):
        super().__init__(max_iter, look_back_window)
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
    
    def is_adjusted(self, dates:list[int], current_date: int) -> int:
        dates = sorted(dates)
        target_index = bisect.bisect_left(dates, current_date)
        if target_index - self.look_back_window - self.interval < 0:
            return -1
        return dates[target_index - self.look_back_window - self.interval]
    

    def _random_tweak(self, distribution: dict[int, float]) -> dict[float, int]:
        while True:
            sample = np.random.normal(self.mean, self.std_dev)
            keys = list(distribution.keys())
            if len(keys) < 2:
                raise ValueError("Agent type < 2!")
            selected_keys = random.sample(keys, 2)
            if distribution[selected_keys[0]] - sample <= 0 or distribution[selected_keys[1]] + sample <= 0:
                continue
            distribution[selected_keys[0]] -= sample
            distribution[selected_keys[1]] += sample
            return distribution
            

    def optimize(self, 
                investment_analyzer: InvestmentAnalyzer,
                dates: list[int],
                start_date: int,
                current_date:int,
                stock_labels: pd.DataFrame,
                stock_pool:pd.DataFrame, 
                distribution: dict[int, float]
                ) -> dict[int, float]:
        dates = sorted(dates)
        current_temp = self.init_temp
        start_index = bisect.bisect_left(dates, start_date)
        search_dates = dates[start_index: start_index + self.look_back_window].copy()
        end_date = search_dates[-1]
        cuurent_distribution = distribution.copy()
        investment_res = self.get_investment_res(investment_analyzer, start_date, current_date, stock_pool, end_date, cuurent_distribution, search_dates)
        current_fitness = self.calcluate_fitness(investment_res=investment_res, stock_labels=stock_labels)
        best_distributions = distribution.copy()
        best_fitness = current_fitness
        for i in tqdm(range(self.max_iter), desc=f"optimizing {current_date}"):

            tweaked_distribution = self._random_tweak(cuurent_distribution.copy())
            fitness = self.calcluate_fitness(investment_res=self.get_investment_res(investment_analyzer, start_date, current_date, stock_pool, end_date, tweaked_distribution, search_dates),
                                              stock_labels=stock_labels)
            if fitness > current_fitness:
                cuurent_distribution = tweaked_distribution
                current_fitness = fitness
            else:
                diff = (current_fitness - fitness) * 10  
                prob = np.exp(-diff / current_temp)  
                if random.random() < prob:  
                    cuurent_distribution = tweaked_distribution
                    current_fitness = fitness
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_distributions = cuurent_distribution
            current_temp *= self.cooling_rate

        return best_distributions


    def get_investment_res(self, investment_analyzer:InvestmentAnalyzer, start_date:int, current_date:int, stock_pool:pd.DataFrame, end_date:int,
                           agent_distributions: dict[int, float], dates: list[int]):
        investment_res = stock_pool[(stock_pool["Date"] >= start_date) & (stock_pool["Date"] <= end_date)].copy()
        def _calc_signal(date: int): 
            current_pool = stock_pool[stock_pool["Date"] == date]["Stock"].tolist()  
            res = investment_analyzer.calculate_stock_disagreement_score(date=date, stock_pool=current_pool, agent_distributions=agent_distributions)  
            return date, res   
        
        date_signal_results = {}  
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  
            date_futures = {executor.submit(_calc_signal, date): date for date in dates}  
            for future in tqdm(  
                concurrent.futures.as_completed(date_futures),  
                total=len(dates),  
                desc=f"Calculating signals {start_date} ~ {end_date}, while current date is {current_date}"  
                ):  
                date, result = future.result()  
                date_signal_results[date] = result 
        
        data_list = []
        for date, res in date_signal_results.items():
            for stock, values in res.items():
                data_list.append([date, stock, values[0], values[1], values[2]])
        result_df = pd.DataFrame(data_list, columns=['Date', 'Stock', 'Signal', 'Signal_mean', 'Signal_std'])
        investment_res = pd.merge(investment_res, result_df, on=['Date', 'Stock'], how='left')
        investment_res[['Signal', 'Signal_mean', 'Signal_std']] = investment_res[['Signal', 'Signal_mean', 'Signal_std']].fillna(0)
        return investment_res

        


        
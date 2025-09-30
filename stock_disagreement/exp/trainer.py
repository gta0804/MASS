from stock_disagreement import Modality, StockDisagreementAgent
import pandas as pd
import numpy as np
from tqdm import tqdm 
from typing import Any, List
import random
from stock_disagreement import InvestmentAnalyzer, BaseOptimizer, SimulatedAnnealingOptimizer
import threading
import concurrent.futures
from scipy.stats import percentileofscore

ROOT_PATH = ""

class StockDisagreementTrainer():
    agent_distributions: dict[int, float] = {}
    date_agent_distributions: dict[int, Any] = {}
    def __init__(self, 
                 num_investor_type: int, 
                 num_agents_per_investor: int,
                 stock_selector_for_per_investor: int,
                 stock_pool: pd.DataFrame,
                 stock_labels: pd.DataFrame,
                 stock_num: int,
                 look_back_window: int,
                 start_date: int | None = None,
                 end_date: int | None = None,
                 use_prev_stock: bool = False,
                 use_self_reflection: bool = False,
                 use_macro_data: bool = False,
                 use_agent_distribution_modification: bool = False,
                 optimizer_look_back_window: int = 2,
                 data_leakage: bool = False,
                 ):
        self.interval = 4
        self.num_investor_type = num_investor_type
        self.stock_selector_for_per_investor = stock_selector_for_per_investor
        self.num_agents_per_investor = num_agents_per_investor
        self.stock_num = stock_num
        self.stock_pool = stock_pool
        self.stock_labels = stock_labels
        self.look_back_window = look_back_window
        self.stock_labels = self.stock_labels.merge(self.stock_pool[["Stock", "Date"]], on=["Stock", "Date"])
        self.optimizer_look_back_window = optimizer_look_back_window
        self.use_agent_distribution_modification = use_agent_distribution_modification

        self.use_prev_stock = use_prev_stock
        if start_date is not None:
            self.start_date = start_date
        else:
            self.start_date = stock_pool["Date"].min()
        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = stock_pool["Date"].max()
        self.stock_pool = self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"] <= self.end_date)]

        self.stock_labels = self.stock_labels[(self.stock_labels["Date"] >= self.start_date) & (self.stock_labels["Date"] <= self.end_date)]
        self.dates = sorted(self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"]<= self.end_date)]["Date"].unique().tolist())
        
        self.agents: List[StockDisagreementAgent] = []
        self.use_self_reflection = use_self_reflection
        self.use_macro_data = use_macro_data
        self.news_info = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/wind-financial-news-info.parq")
        self.news_relationship = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/wind-financial-news-relationship.parq")
        self.news_info["Date"] = self.news_info["Date"].astype("int32")
        self.news_relationship["Date"] = self.news_relationship["Date"].astype("int32")
        self._init_agents() 
        self.data_leakage = data_leakage
        self.optimzer = SimulatedAnnealingOptimizer(look_back_window=optimizer_look_back_window)

    def _init_agents(self,) -> None:

        def cal_pe_quantile(data: pd.DataFrame, look_back_year: int = 10) -> pd.DataFrame:
            data_copy = data.copy()
            data_copy["dt"] = pd.to_datetime(data_copy["Date"].astype(str), format="%Y%m%d")
            data_copy.sort_values("dt", inplace=True)
            data_copy.reset_index(drop=True, inplace=True)  

            min_periods = look_back_year * 250
            data_copy["quantile"] = data_copy["Value"].rolling( window= min_periods, closed="both").apply(lambda x: percentileofscore(x, x.iloc[-1]), raw=False)    
            data_copy["quantile"] = data_copy["quantile"].round(1)   
            data_copy["Date"] = data_copy["dt"].dt.strftime("%Y%m%d").astype(int)  
            return data_copy[["Date", "Value", "quantile"]]  


        loan_rate = pd.read_csv(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/macro_data/China_1-Year_Loan_Prime_Rate_LPR.csv")
        cpi = pd.read_csv(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/macro_data/China_CPI_YoY_Current_Month.csv")
        csi_300_pe = pd.read_csv(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/macro_data/csi_300_pe_ttm.csv")
        csi_300_pe  = cal_pe_quantile(csi_300_pe)
        market_sentiment_index = pd.read_csv(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/macro_data/Market_Sentiment_Index.csv")
        yield_on_China_bonds = pd.read_csv(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/macro_data/yield_on_China_10_year_government_bonds.csv") 
        for num in tqdm(range(self.num_investor_type), desc="Init agents"):
            modalities = list(Modality)
            selected_modalities = random.sample(modalities, k=random.randint(1, 3))  
            result = Modality(0)  
            for modality in selected_modalities:
                result |= modality  
            for i in range(self.num_agents_per_investor):
                current_agent = StockDisagreementAgent(stock_num= self.stock_num, 
                                                   stock_pool =self.stock_pool,
                                                   stock_labels = self.stock_labels,
                                                   is_self_reflective =True,
                                                   max_reflective_times =10, 
                                                   start_date = self.start_date,
                                                   end_date =self.end_date,
                                                   modality = result,
                                                   use_prev_stock = self.use_prev_stock,
                                                   cpi = cpi,
                                                   csi_300_pe = csi_300_pe,
                                                   loan_rate = loan_rate,
                                                   market_sentiment_index = market_sentiment_index,
                                                   yield_on_China_bonds = yield_on_China_bonds,
                                                   use_self_reflection = self.use_self_reflection,
                                                   use_macro_data = self.use_macro_data) 
                current_agent.prepare_data_source(self.news_info, self.news_relationship)
                # if not self.use_macro_data:
                #     current_agent.generate_strategy_and_stock_selector()
                # else:
                #     current_agent.generate_strategy_and_stock_selector(self.start_date)
                self.agent_distributions[result] = 1.0
                self.agents.append(current_agent) 


    
    def run(self) -> pd.DataFrame:  
        def _process_agent(agent:StockDisagreementAgent, date: int, stock_selector):  
            agent.invest(date, stock_selector)
        
        if not self.use_agent_distribution_modification:
            total_agent_tasks = len(self.dates) * len(self.agents) 
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  
                futures = [  
                    executor.submit(_process_agent, agent, date, self.stock_selector_for_per_investor)  
                    for date in self.dates  
                    for agent in self.agents  
                ]    
                for future in tqdm(  
                    concurrent.futures.as_completed(futures),  
                    total=total_agent_tasks,  
                    desc="Processing agents"  
                ):   
                    future.result() 
        else:
            for date in self.dates:
                total_tasks = len(self.agents)
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(_process_agent, agent, date, self.stock_selector_for_per_investor)
                        for agent in self.agents
                    ]
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=total_tasks,
                        desc= f"Processing agent on {date}"
                    ):
                        future.result()
                res = self.optimzer.is_adjusted(self.dates, date)
                if res > 0:
                    self.date_agent_distributions[date] = self.agent_distributions.copy()
                    best_distributions = self.optimzer.optimize(investment_analyzer=self.agents[0].investment_analyzer,
                                                                dates = self.dates,
                                                                start_date=res,
                                                                current_date= date,
                                                                stock_labels= self.stock_labels,
                                                                stock_pool= self.stock_pool,
                                                                distribution=self.agent_distributions)
                    self.agent_distributions = best_distributions
                    if self.data_leakage:
                        self.date_agent_distributions[date] = best_distributions
                else:
                    self.date_agent_distributions[date] = self.agent_distributions
                
        investment_analyzer = self.agents[0].investment_analyzer  
        investment_res = self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"] <= self.end_date) ].copy()  


        def _calc_signal(date: int, agent_distributions: dict[int, float]): 
            current_pool = self.stock_pool[self.stock_pool["Date"] == date]["Stock"].tolist()  
            res = investment_analyzer.calculate_stock_disagreement_score(date, current_pool, agent_distributions=agent_distributions)  
            return date, res   
        
        date_signal_results = {}  
        if not self.use_agent_distribution_modification:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  
                date_futures = {executor.submit(_calc_signal, date, self.agent_distributions): date for date in self.dates}  
                for future in tqdm(  
                    concurrent.futures.as_completed(date_futures),  
                    total=len(self.dates),  
                    desc="Calculating signals"  
                    ):  
                    date, result = future.result()  
                    date_signal_results[date] = result 
        else:
            for date in tqdm(self.dates, desc = "calc investment res in optimzing mode."):
                dt, res = _calc_signal(date, self.date_agent_distributions[date])
                date_signal_results[dt] = res

        data_list = []
        for date, res in date_signal_results.items():
            for stock, values in res.items():
                data_list.append([date, stock, values[0], values[1], values[2]])
        result_df = pd.DataFrame(data_list, columns=['Date', 'Stock', 'Signal', 'Signal_mean', 'Signal_std'])
        investment_res = pd.merge(investment_res, result_df, on=['Date', 'Stock'], how='left')
        for date, res in date_signal_results.items():  
            for stock, values in res.items():  
                indexer = (investment_res["Date"] == date) & (investment_res["Stock"] == stock)  
                investment_res.loc[indexer, "Signal"] = values[0]  
                investment_res.loc[indexer, "Signal_mean"] = values[1]  
                investment_res.loc[indexer, "Signal_std"] = values[2]  
        return investment_res

    # def run(self) -> pd.DataFrame: 
    #     def _process_agent(agent:StockDisagreementAgent, date: int, stock_selector):  
    #         agent.stock_selector.get_next()  
    #         agent.invest(date, stock_selector)  
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  
    #         for date in self.dates:  
    #             futures = []  
    #             for agent in self.agents:  
    #                 future = executor.submit(  
    #                 _process_agent,  
    #                 agent,  
    #                 date,  
    #                 self.stock_selector_for_per_investor)  
    #                 futures.append(future)
    #             for future in concurrent.futures.as_completed(futures):  
    #                 future.result() 
    #     investment_analyzer = self.agents[0].investment_analyzer
    #     investment_res = self.stock_pool.copy()
    #     investment_res["Signal"] = 0
    #     investment_res["Signal_mean"] = 0
    #     investment_res["Signal_std"] = 0
    #     for date in self.dates:
    #         current_pool = self.stock_pool[self.stock_pool["Date"] == date]["Stock"].tolist()
    #         res = investment_analyzer.calculate_stock_disagreement_score(date, current_pool)
    #         for stock in res: 
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal"  
    #             ] = res[stock][0]
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal_mean"  
    #             ] = res[stock][1]
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal_std"  
    #             ] = res[stock][2]
    #     return investment_res

    # def run(self, ) -> InvestmentAnalyzer:
    #     for date in self.dates:
    #         for agent in self.agents:
    #             agent.stock_selector.get_next()
    #             agent.invest(date, self.stock_selector_for_per_investor)
        
    #     return self.agents[0].investment_analyzer      

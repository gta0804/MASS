from stock_disagreement import OpenAIModel
from openai import OpenAI
from typing import *
from stock_disagreement.agent import RandomStockSelector, BasicStockSelector, IndustryBasisStockSelector, IndustryEqualStockSelector, MVEqualStockSelector
from stock_disagreement.prompts import (INVESTING_DECISION_EXAMPLE, 
                                        INVESTING_DECISION_INSTRUCTION, 
                                        INVESTING_STYLE_INSTRUCTION,
                                        INVSETING_STYLE_EXAMPLE, 
                                        INVESTING_DECISION_INSTRUCTION_USING_SELF_REFLECTION,
                                        INVESTING_STYLE_INSTRUCTION_WITH_MARCO_DATA,
                                        INVESTING_STYLE_WITH_MACRO_DATA_EXAMPLE)

from stock_disagreement.agent.investment_analyzer import InvestmentAnalyzer
from stock_disagreement.agent.investing_history import InvestingHistory
import pandas as pd
import numpy as np
import json
from enum import IntFlag 
from scipy.stats import percentileofscore  

MAX_RETRIRES = 3
ROOT_PATH = ""

def calculate_pe_quantile_5y(df:pd.DataFrame) -> pd.DataFrame:  
    

    df_copy = df.copy()  
    df_copy["dt"] = pd.to_datetime(df_copy["Date"].astype(str), format="%Y%m%d")   
    df_copy.sort_values("dt", inplace=True)  
    df_copy.reset_index(drop=True, inplace=True)  
    
    min_periods = 10 * 250  
    df_copy["quantile"] = df_copy["Value"].rolling(  
        window= min_periods, closed="both"  
    ).apply(lambda x: percentileofscore(x, x.iloc[-1]), raw=False)  
    
    df_copy["quantile"] = df_copy["quantile"].round(1)  
    
    df_copy["Date"] = df_copy["dt"].dt.strftime("%Y%m%d").astype(int)  
    return df_copy[["Date", "Value", "quantile"]]  

class Modality(IntFlag):  
    FUDAMENTAL_VALUTION = 0b00000001  
    FUDAMENTAL_DIVIDEND = 0b00000010 
    FUDAMENTAL_GROWTH = 0b000000100
    FUDAMENTAL_QUALITY = 0b000001000
    NEWS = 0b000010000      
    BASE_DATA = 0b000100000  
    CROSS_INDUSTRY_LABEL = 0b001000000
    RISK_FACTOR = 0b010000000
    PRICE_FEATURE = 0b100000000 




class StockDisagreementAgent():
    model_name: str = "Qwen2.5-72B-Instruct"
    model_server: str = "http://127.0.0.1:1025/v1"
    api_key: str = ""
    prev_stock: list[str] | None = None
    system_prompt: str = "You are a helpful assistant. Strictly follow the user's input prompt and output the result in JSON format as specified. Do not include any additional content."
    investing_history: InvestingHistory = InvestingHistory()

    def __init__(self, 
                 stock_num: int, 
                 stock_pool: pd.DataFrame,
                 stock_labels: pd.DataFrame,
                 csi_300_pe: pd.DataFrame,
                 cpi: pd.DataFrame,
                 loan_rate: pd.DataFrame,
                 yield_on_China_bonds: pd.DataFrame,
                 market_sentiment_index: pd.DataFrame,
                 is_self_reflective: bool = False,
                 max_reflective_times: int = 10,
                 start_date: int | None = None,
                 end_date: int | None = None,
                 modality: Modality = Modality.FUDAMENTAL_VALUTION,
                 use_prev_stock: bool = False,
                 use_self_reflection: bool = False,
                 use_macro_data: bool = False,
                 ):
        self.client = OpenAI(api_key=self.api_key, base_url=self.model_server)
        self.model = OpenAIModel(self.model_name, None, 80000)
        self.stock_num = stock_num
        self.stock_pool = stock_pool
        self.stock_labels = stock_labels
        self.is_self_reflective = is_self_reflective
        self.max_reflective_times = max_reflective_times
        self.loan_rate = loan_rate
        self.cpi = cpi
        self.market_sentiment_index = market_sentiment_index
        self.csi_300_pe = csi_300_pe
        self.yield_on_China_bonds = yield_on_China_bonds
        self.csi_300_pe = calculate_pe_quantile_5y(csi_300_pe)

        self.csi_300_pe["Date"] = self.csi_300_pe["Date"].astype("int32")

        self.cpi["Date"] = self.cpi["Date"].astype("int32")

        self.yield_on_China_bonds["Date"] = pd.to_datetime(self.yield_on_China_bonds["Date"].astype(str))
        self.yield_on_China_bonds["Date"] = self.yield_on_China_bonds["Date"].dt.strftime("%Y%m%d").astype("int32")
        self.yield_on_China_bonds = self.yield_on_China_bonds.sort_values("Date")
        self.yield_on_China_bonds["1_day_diff"] = self.yield_on_China_bonds["Value"].diff(periods=1)
        self.yield_on_China_bonds["30_day_diff"] = self.yield_on_China_bonds["Value"].diff(periods=30)
        self.yield_on_China_bonds["180_day_diff"] = self.yield_on_China_bonds["Value"].diff(periods=180)

        self.loan_rate["Date"] = self.loan_rate["Date"].astype("int32")

        self.market_sentiment_index["Date"] = self.market_sentiment_index["Date"].astype("int32")


        if start_date is None:
            self.start_date = stock_pool["Date"].min()
        else:
            self.start_date = start_date
        if end_date is None:
            self.end_date = stock_pool["Date"].max()
        else:
            self.end_date = end_date
        self.modality = modality
        self.prepare_data: dict[str, pd.DataFrame] = {}
        self.description: dict[str, str] = {}
        self.strategy_input: str = ""
        # self.prepare_data_source()
        self.strategy: dict[str, Any] = None
        self.stock_selector: BasicStockSelector = None
        self.use_macro_data = use_macro_data
        self.investment_analyzer = InvestmentAnalyzer()
        self.use_prev_stock = use_prev_stock
        self.use_self_reflection = use_self_reflection 

    def generate_macro_data_input(self, date: int) -> str:
        latest_loan_rate_date = self.loan_rate[self.loan_rate["Date"] <= date]["Date"].max()
        latest_loan_rate = self.loan_rate[self.loan_rate["Date"] == latest_loan_rate_date]["Value"].iloc[0]
        res = ""
        res += f"The latest 1 year loan prime rate is {str(latest_loan_rate)}. "
        
        latest_cpi_rate_date = self.cpi[self.cpi["Date"] <= date]["Date"].max()
        latest_cpi = self.cpi[self.cpi["Date"] == latest_cpi_rate_date]["Value"].iloc[0]
        res += f"The latest month China CPI YOY growth rate is {str(latest_cpi)}. "
        latest_yield_on_China_bonds_date = self.yield_on_China_bonds[self.yield_on_China_bonds["Date"] <= date]["Date"].max()
        latest_yield = self.yield_on_China_bonds[self.yield_on_China_bonds["Date"] == latest_yield_on_China_bonds_date]["Value"].iloc[0]
        one_day_yield_diff = self.yield_on_China_bonds[self.yield_on_China_bonds["Date"] == latest_yield_on_China_bonds_date]["1_day_diff"].iloc[0]
        month_yield_diff = self.yield_on_China_bonds[self.yield_on_China_bonds["Date"] == latest_yield_on_China_bonds_date]["30_day_diff"].iloc[0]
        half_year_yield_diff = self.yield_on_China_bonds[self.yield_on_China_bonds["Date"] == latest_yield_on_China_bonds_date]["180_day_diff"].iloc[0]
        res += f"The latest yield of China ten year government bonds is {str(latest_yield)}%, while yield increases {str(round(one_day_yield_diff * 100))} BP over past one day,\
         increases {str(round(month_yield_diff * 100))} BP over past one month, increases {str(round(half_year_yield_diff * 100))} BP over past half an year. "
        latest_pe_date = self.csi_300_pe[self.csi_300_pe["Date"] <= date]["Date"].max()
        latest_pe = self.csi_300_pe[self.csi_300_pe["Date"] == latest_pe_date]["Value"].iloc[0]
        latest_pe_quantile = self.csi_300_pe[self.csi_300_pe["Date"] == latest_pe_date]["quantile"].iloc[0]
        res += f"The latest csi_300 pe is {str(latest_pe)}, and the current PE ratio of the CSI 300 is at the {latest_pe_quantile} percentile over the past 5 years(0 indicates most undervalued, and 100 indicates most overvalued). "
        market_sentiment_index_date = self.market_sentiment_index[self.market_sentiment_index["Date"] <= date]["Date"].max()
        market_sentiment_index_pricechange = self.market_sentiment_index[self.market_sentiment_index["Date"] == market_sentiment_index_date]["PriceChange"].iloc[0]
        res += f"The latest market sentiement index got {str(round( market_sentiment_index_pricechange * 100, 2))} % return."
        
        return res
        
   
    def prepare_data_source(self, news_info: pd.DataFrame, news_relationship: pd.DataFrame) -> None:
        if self.modality & Modality.FUDAMENTAL_VALUTION:
            keys = ["E/P", "B/P", "CF/P", "S/P",
                    "Log-orthogonalized E/P", "Log-orthogonalized B/P",
                    "Log-orthogonalized CF/P", "Log-orthogonalized S/P", "EBITDA/EV"]
            
            self.prepare_data["fudamental_valuation"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq")[["Stock", "Date"] + keys]
            self.description["E/P"] = "The inverse of the P/E ratio (E/P) indicates the earnings yield, showing the percentage of profit generated per dollar invested in the stock."
            self.description["B/P"] = "Inverse of P/B (B/P) indicates the book yield, showing the return on book value per dollar invested."
            self.description["S/P"] = "Inverse of P/S (S/P) reflects the sales yield, showing sales generated per dollar invested."
            self.description["CF/P"] = "Inverse of P/CF (CF/P) shows the cash flow yield, representing cash flow generated per dollar invested."
            self.description["Log-orthogonalized E/P"] = "Log-orthogonalized version of E/P, removing some kind of cap basis.Log-orthogonalized version of E/P, removing some kind of cap basis."
            self.description["Log-orthogonalized B/P"] = "Log-orthogonalized version of B/P, removing some kind of cap basis."
            self.description["Log-orthogonalized CF/P"] = "Log-orthogonalized version of CF/P, removing some kind of cap basis."
            self.description["Log-orthogonalized S/P"] = "Log-orthogonalized version of S/P, removing some kind of cap basis."
            self.description["EBITDA/EV"] = "Measures a company's return on enterprise value, indicating operating earnings (EBITDA) generated per dollar of EV."
        
        if self.modality & Modality.FUDAMENTAL_QUALITY:
            keys = ["ROE stability", "ROA stability", "ROE", "Annualized ROE"]
            self.prepare_data["fudamental_quality"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq")[["Stock", "Date"] + keys]
            self.description["ROE"] = "ROE Measures profitability, showing net income generated per dollar of shareholders' equity."
            self.description["ROE stability"] = "TS_Mean(ROE, 8) / TS_Std(ROE, 8), measuring both absolute value and stability of ROE."
            self.description["ROA stability"] = "TS_Mean(ROA, 8) / TS_Std(ROA, 8), measuring both absolute value and stability of ROA."
            self.description["Annualized ROE"] = "Annualized version of ROE."
        
        if self.modality & Modality.FUDAMENTAL_DIVIDEND:
            keys = ["Dividend yield", "Log-orthogonalized dividend yield", "Log-orthogonalized dividend yield", "Dividend yield incl repo & mjrholder trans"]
            self.prepare_data["fudamental_dividend"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq")[["Stock", "Date"] + keys]
            self.description["Dividend yield"] = "Dividend yield indicates annual dividends received per dollar invested, expressed as a percentage of the stock price"
            self.description["Log-orthogonalized dividend yield"] = "Log-orthogonalized version of dividend yield, removing some kind of cap basis."
            self.description["Dividend yield incl repo & mjrholder trans"] = "Dividend yield including stock repurchasing and major holder trading."
        
        if self.modality & Modality.FUDAMENTAL_GROWTH:
            keys = ["Revenue TTM YoY growth rate", "Net profit TTM YoY growth rate", "Non-GAAP net profit YoY growth rate"]
            self.prepare_data["fudamental_growth"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq")[["Stock", "Date"] + keys]
            self.description["Revenue TTM YoY growth rate"] = "Measures the percentage change in trailing twelve months' revenue compared to the same period last year."
            self.description["Net profit TTM YoY growth rate"] = "Measures the percentage change in trailing twelve months' net profit compared to the same period last year."
            self.description["Non-GAAP net profit YoY growth rate"] = "Indicates the percentage change in non-GAAP net profit compared to the same period last year."
        
        if self.modality & Modality.RISK_FACTOR:
            keys = ["Intraday volatility", "Liquidity", "Residual volatility"]
            self.prepare_data["risk_factor"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/sub_fudamental_data.parq")[["Stock", "Date"] + keys]
            self.description["Intraday volatility"] = "Measuring the price fluctuation range of a stock within a single trading day."
            self.description["Liquidity"] = "Weighted average of monthly, quarterly and yearly turnover ratio."
            self.description["Residual volatility"] = "Residual volatility measures the unexplained variability in a security's returns after accounting for market or factor influences, indicating idiosyncratic risk."

        
        if self.modality & Modality.BASE_DATA:
            keys = ["Open", "High", "Low", "Close", "Value"]
            self.prepare_data["base_data"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/base_data.parq")[["Stock", "Date"] + keys]
        
        if self.modality & Modality.CROSS_INDUSTRY_LABEL:
            keys = ["Industry", "Daily_Return"]
            self.prepare_data["cross_industry_label"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/industry_ret.parq")[keys]
            self.prepare_data["stock_basic_data"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/stock_basic_data.parq")
            self.description["Daily_Return"] = "One-day return of holding the sector's constituent stocks."
        
        if self.modality & Modality.PRICE_FEATURE:
            keys = ["price_value_feature_0", "price_value_feature_1", "price_value_feature_2",
                    "price_value_feature_3", "price_value_feature_4", "price_value_feature_5"]
            
            self.prepare_data["price_value_data"] = pd.read_parquet(f"{ROOT_PATH}/stock_prediction_benchmark/stock_disagreement/dataset/price_feature.parq")[["Stock", "Date"] + keys]
            self.description["price_value_feature_0"] = "Combined feature using price and value data, theoretically negatively correlated with future daily returns over the next 1-5 days."
            self.description["price_value_feature_1"] = "Combined feature using price and value data, theoretically negatively correlated with future daily returns over the next 1-5 days."
            self.description["price_value_feature_2"] = "Combined feature using price and value data, theoretically negatively correlated with future daily returns over the next 1-5 days."
            self.description["price_value_feature_3"] = "Combined feature using price and value data, theoretically positively correlated with future daily returns over the next 1-5 days."
            self.description["price_value_feature_4"] = "Combined feature using price and value data, theoretically negatively correlated with future daily returns over the next 1-5 days."
            self.description["price_value_feature_5"] = "Combined feature using price and value data, theoretically possitively correlated with future daily returns over the next 1-5 days."
        
        if self.modality & Modality.NEWS:
            news_info = news_info
            news_relationship = news_relationship.merge(self.stock_pool[["Stock", "Date"]], on=["Stock", "Date"])
            self.prepare_data["news"] = news_info[["Date", "NewsId", "NewsTitle"]].merge(news_relationship[["Stock","NewsId"]], on=["NewsId"])
            # self.prepare_data["news"] = news_info.iloc[:, ~self.prepare_data["news"].columns.isin(["NewsId"])]


    def generate_strategy_input(self, date: int = None) -> tuple[str, str]: 
        def generate_headers() -> dict[str, str]:
            res = {}
            for key, description in self.description.items():
                res[key] = description
            if "news" in self.prepare_data.keys():
                res["news"] = f"Investors read news to make investment descisions. The example news is: {self.prepare_data['news']['NewsTitle'].head(3).to_list()}"
            return res
        headers = generate_headers()
        input = f"Investors read following information to make investment decisions. The information is in key-value format, key representing name, value representing descriptions.: \
                 {str(headers)}. \n"
        if date is None:
            return input, None
        macro_input = self.generate_macro_data_input(date) 
        return input, macro_input
    
    def genererate_self_reflection_input(self, date:int) -> dict[str, Any]:
        def get_stock_labels(date: int, x: int, stock_labels: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
            if x <= 0:
                raise ValueError(f"{x} is expected to be positive integers.")
            stock_labels = stock_labels.sort_values("Date")
            sorted_dates = np.sort(stock_labels["Date"].unique())
            idx = np.searchsorted(sorted_dates, date, side="left")
            target_idx = idx - x
            if target_idx < 0:
                return False, None
            target_date = sorted_dates[target_idx]
            target_data = stock_labels[stock_labels["Date"] == target_date]
            success = len(target_data) > 0
            return success, target_data
        res_dict = {}
        pass_days = [1, 5, 10]
        for pass_day in pass_days:
            success, pass_stock_labels = get_stock_labels(date, pass_day, self.stock_labels)
            if not success:
                break
            success, pass_stock_list = self.investing_history.get_history_stocks(pass_day)
            if not success:
                break
            success, chosen_stock_list = self.investing_history.get_chosen_stocks(pass_day)
            if not success:
                break
            success, pass_investing_history = self.investing_history.get_record(pass_day)
            if not success:
                break
            current_dict = {}
            current_dict["description"] = f"Investing history {pass_day} ago."
            current_dict["input_data"] = pass_investing_history
            sub_stock_label = pass_stock_labels[pass_stock_labels["Stock"].isin(pass_stock_list)].copy()
            sub_stock_label["rank"] = sub_stock_label[f"{pass_day}_15_labelB"].rank(ascending=False, method="min")
            investment_res = ""
            for stock in chosen_stock_list:
                if stock in pass_stock_list:
                    if stock in sub_stock_label["Stock"].unique().tolist():
                        investment_res += f"For chosen stock {stock}, you get rank {sub_stock_label[sub_stock_label['Stock'] == stock]['rank'].iloc[0].astype(int)} out of {len(pass_stock_list)}."
            current_dict["investment_res"] = investment_res
            res_dict[f"{pass_day} ago"] = current_dict
        return res_dict

    def generate_strategy_and_stock_selector(self, date: int = None) -> None:
        data_input, macro_input = self.generate_strategy_input(date)
        if not self.use_macro_data:
            strategy_input = INVESTING_STYLE_INSTRUCTION.format(examples=INVSETING_STYLE_EXAMPLE, input_data=data_input)
        else:
            strategy_input = INVESTING_STYLE_INSTRUCTION_WITH_MARCO_DATA.format(examples=INVESTING_STYLE_WITH_MACRO_DATA_EXAMPLE,
                                                                                input_data=data_input,
                                                                                macro_data=macro_input)
        strategy,_ = self.model.chat_generate(client=self.client,
                                       system_prompt=self.system_prompt,
                                       input_string=strategy_input
                                       )
        json_strategy = json.loads(strategy)
        selector_name = json_strategy["Details"]["StockPoolSelector"]
        if isinstance(selector_name, str):
            if selector_name == "RandomStockSelector":
                stock_selector = RandomStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date)
            elif selector_name == "IndustryEqualStockSelector":
                stock_selector = IndustryEqualStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date)
            elif selector_name == "MVEqualStockSelector":
                stock_selector = MVEqualStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date)
            else:
                stock_selector = RandomStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date)
        # elif isinstance(selector_name, dict):
        #     assert len(list(selector_name.keys())) == 1
        #     assert list(selector_name.keys())[0] == "I"
        #     basis_indutsry = selector_name["IndustryBasisStockSelector"]
        #     assert isinstance(basis_indutsry, list[str])
        #     stock_selector = IndustryBasisStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date, basis_indutsry)
        else:
            print("stock selector does not match, using default random stock selector.")
            stock_selector = RandomStockSelector(self.stock_num, self.stock_pool, self.start_date, self.end_date)
        assert isinstance(stock_selector, BasicStockSelector)
        if self.stock_selector is None:
            self.stock_selector = stock_selector
        self.strategy = json_strategy
  
    def invest(self, date: int, num_stocks: int) -> None:
        # if self.use_macro_data:
        self.generate_strategy_and_stock_selector(date)
        # if self.use_prev_stock:
        #     if self.prev_stock is None:
        #         current_stock = self.stock_selector.select_stock_for_llm(date=date)
        #     else:
        #         all_in_stock_pool = True
        #         for stock in self.prev_stock:
        #             if stock not in self.stock_pool[self.stock_pool["Date"] == date]["Stock"].tolist():
        #                 all_in_stock_pool = False
        #                 break
        #         if all_in_stock_pool:
        #             current_stock = self.prev_stock
        #         else:
        #             current_stock = self.stock_selector.select_stock_for_llm(date=date)
        # else:
        #     current_stock = self.stock_selector.select_stock_for_llm(date=date)
        # self.prev_stock = current_stock

        current_stock = self.stock_selector.select_stock_for_llm(date=date)

        prepare_datas = {}
        descriptions = {}
        for key in self.prepare_data:
            if "Stock" in self.prepare_data[key].columns and "Date" in self.prepare_data[key].columns:
                prepare_datas[key] = self.prepare_data[key][(self.prepare_data[key]["Date"] == date) & (self.prepare_data[key]["Stock"].isin(current_stock))]
            elif "Stock" in self.prepare_data[key].columns:
                prepare_datas[key] = self.prepare_data[key][(self.prepare_data[key]["Stock"].isin(current_stock))]
        for key in self.description:
            descriptions[key] = self.description[key]
        input_data = f"Input Data for investing decision:1. descriptions: {str(descriptions)}, 2. input data: {str(prepare_datas)}."
        investment_strategy = f" Investment strategy: {str(self.strategy)}"
        if not self.use_self_reflection:
            llm_input = INVESTING_DECISION_INSTRUCTION.format(num_stocks = num_stocks, examples = INVESTING_DECISION_EXAMPLE, input_data = input_data + investment_strategy)
        else:
            reflection_input = self.genererate_self_reflection_input(date)
            llm_input = INVESTING_DECISION_INSTRUCTION_USING_SELF_REFLECTION.format(num_stocks = num_stocks, 
                                                                                    examples = INVESTING_DECISION_EXAMPLE,                                                                            input_data = input_data + investment_strategy,
                                                                                 decision_history = str(reflection_input)) 
        for attempt in range(MAX_RETRIRES):
            try:
                res, reason = self.model.chat_generate(client=self.client,
                                    system_prompt=self.system_prompt,
                                    input_string=llm_input)
                selected_stocks = json.loads(res)["Stock"]
                print(f"on {date}, agent {self.modality} chooses {str(selected_stocks)}")
                # if num_stocks != len(selected_stocks):
                #     raise ValueError(f"LLM does not output required num stocks, expected {num_stocks}, actual {len(selected_stocks)}")
                for stock in selected_stocks:
                    if stock not in current_stock:
                        print(f"current stock is {str(current_stock)}, whereas current selected stock is {stock}")
                        raise ValueError(f"LLM output illegal stock code {stock}!")
                break
            except Exception as e:
                self.investing_history.add_records(input_data, current_stock, [])
                print(f"occuring exceptions {str(e)} in investment decisions")
                if attempt >= MAX_RETRIRES - 1:
                    print("Exceeding max retries, giving up.")
                    return
        self.investing_history.add_records(input_data, current_stock, selected_stocks)
        selected_stock_res = {}
        for stock in current_stock:
            if stock in selected_stocks:
                selected_stock_res[stock] = 1
            else:
                selected_stock_res[stock] = 0
        self.investment_analyzer.record_score(date, self.modality, 1, selected_stock_res)

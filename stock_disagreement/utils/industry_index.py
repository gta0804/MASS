"""简单合成行业指数"""
import pandas as pd
import numpy as np

class IndustryIndex():
    def __init__(self, ):
        pass

    def __call__(self, base_data: pd.DataFrame,
                 industry_data: pd.DataFrame,
                 label: pd.DataFrame) -> pd.DataFrame:
        """合成行业指数
        base_data: Stock Date FREE_MV 用每日的流通市值加权
        industry_data: Stock Industry 包含股票所属行业
        label: Stock Date 1_15_labelB 包含所有标的每日涨幅
        """
        industry_data = industry_data.query("Industry != '-'").copy()
        industry = industry_data.drop_duplicates(["Industry"])
        base_data_with_industry = base_data.merge(industry_data, on=["Stock"], how="inner")
        base_data_with_industry["ratio"] = base_data_with_industry["FREE_MV"] / base_data_with_industry.groupby(["Date", "Industry"])["FREE_MV"].transform("sum")
        base_data_with_label = base_data_with_industry.merge(label[["Stock", "Date", "1_15_labelB"]], on=["Stock", "Date"])
        base_data_with_label["weighted_return"] = base_data_with_label["1_15_labelB"] * base_data_with_label["ratio"]
        industry_daily_return = base_data_with_label.groupby(["Date", "Industry"])["weighted_return"].sum().reset_index()
        
        # 重命名列
        industry_daily_return.rename(columns={"weighted_return": "Daily_Return"}, inplace=True)
        
        # 合并 Label 字段
        industry_daily_return = industry_daily_return.merge(industry_data[['Industry', 'Label']], on=["Industry"], how="left")
        
        # 确保 Label 字段不重复
        industry_daily_return.drop_duplicates(inplace=True)
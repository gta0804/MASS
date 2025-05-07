import threading
import numpy as np

class InvestmentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.data = {}
        return cls._instance
    
    def __init__(self):  
        if not hasattr(self, 'initialized'):  
            self.initialized = True 
            self.data = {}  
    
    def record_score(self, date:int, investor_type: int, investor_id: int, stocks: dict[str, int]):
        with threading.Lock():
            if date not in self.data:
                self.data[date] = {}
            if investor_type not in self.data[date]:
                self.data[date][investor_type] = {}
            for (key, value) in stocks.items():
                if key not in self.data[date][investor_type]:
                    self.data[date][investor_type][key] = {
                        "score": 0,
                        "num_investors": 0
                    }
                self.data[date][investor_type][key]["score"] += value
                self.data[date][investor_type][key]["num_investors"] += 1
    
    def calculate_stock_disagreement_score(self,
                                           date: int,
                                           stock_pool: list[str], 
                                           agent_distributions:dict[int, float] ,
                                           alpha:float = 0.5):
        res = {}
        distributions = []
        with threading.Lock():
            for investor_type in self.data[date]:
                if investor_type in agent_distributions:
                    distribution = agent_distributions[investor_type]
                else:
                    distribution = 0.0
                    raise ValueError(f"agent type {investor_type} does not exist!")
                for stock in stock_pool:
                    if stock not in self.data[date][investor_type]:
                        self.data[date][investor_type][stock] = {
                            "score": 0,
                            "num_investors": 0
                    }
                for stock_code in self.data[date][investor_type]:
                    stock_data = self.data[date][investor_type][stock_code]
                    total_scores = stock_data["score"]
                    total_investors = stock_data["num_investors"]
                    if stock_code not in res:
                        res[stock_code] = [(total_scores / total_investors) if total_investors != 0 else 0]
                    else:
                        res[stock_code].append((total_scores / total_investors)if total_investors != 0 else 0)
                distributions.append(distribution)
        for stock_code in res:
            mean_value = np.sum(np.array(res[stock_code]) * distributions) / np.sum(distributions)
            std = np.sqrt(np.average((np.array(res[stock_code]) - mean_value) ** 2, weights=distributions))
            res[stock_code] = [alpha * mean_value - (1 - alpha) * std, mean_value, -std]
        return res


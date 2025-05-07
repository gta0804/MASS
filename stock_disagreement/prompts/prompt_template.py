INVESTING_STYLE_INSTRUCTION = """Give following input data:
1.Input time-series data column name and their descriptions in json format(textual data example).
Please try to analyze and summarize an abstract investing style description. 
The output format is a json. The specific format of the output JSON is:
{{
   "Outline": "The outline and general description for investment style within 50 words. The outline is a summarization, without any details below.",  
   "Details": {{
      "Risk Appetite": "conservative | moderate | moderately conservative | moderately aggressive | aggressive",  
      "Holding Period": "one day | about one week | about one month | about half a year | more than one year",  
      "Strategy Consistency": "[0, 1] (Refers to the investor's ability to adhere to and execute their investment strategy with persistence and coherence, regardless of short-term market fluctuations or emotional influences. Higher number means high consistency)",  
      "Rationality": "[0, 1] (Refers to whether the investor's decision-making process is based on logic, data, and long-term objectives rather than emotions, biases, or short-term market noise. Higher number means high rationality)",  
      "StockPoolSelector": "Specify what kind of preference you'd like to construct your watchlist stocks. The possible preferences are: 1. RandomStockSelector: Randomly construct your watchlist. 2. IndustryEqualStockSelector: Construct a stock pool with balanced distribution across industries. 3. MVEqualStockSelector: Construct a stock pool with balanced distribution across market capitalizations. 4. IndustryBasisStockSelector: Prefer stocks from specific industries and output the preferred industries. The result is presented in a list format.",
   }}  
}}
{examples}
(END_OF_EXAMPLES)
Input data:
{input_data}
Your investing style:
"""

INVESTING_STYLE_INSTRUCTION_WITH_MARCO_DATA = """
Give following input data:
1. Input time-series data column name and their descriptions in json format(textual data example).
2. latest macroeconomic and market insights.
Please try to analyze and summarize an abstract investing style description. 
The output format is a json. The specific format of the output JSON is:
{{
   "Outline": "The outline and general description for investment style within 50 words. The outline is a summarization about your investing strategy and your insights into the subsequent trend of the stock market, without any details below.",  
   "Details": {{
      "Risk Appetite": "conservative | moderate | moderately conservative | moderately aggressive | aggressive",  
      "Holding Period": "one day | about one week | about one month | about half a year | more than one year",  
      "Strategy Consistency": "[0, 1] (Refers to the investor's ability to adhere to and execute their investment strategy with persistence and coherence, regardless of short-term market fluctuations or emotional influences. Higher number means high consistency)",  
      "Rationality": "[0, 1] (Refers to whether the investor's decision-making process is based on logic, data, and long-term objectives rather than emotions, biases, or short-term market noise. Higher number means high rationality)",  
      "StockPoolSelector": "Specify what kind of preference you'd like to construct your watchlist stocks. The possible preferences are: 1. RandomStockSelector: Randomly construct your watchlist. 2. IndustryEqualStockSelector: Construct a stock pool with balanced distribution across industries. 3. MVEqualStockSelector: Construct a stock pool with balanced distribution across market capitalizations. 4. IndustryBasisStockSelector: Prefer stocks from specific industries and output the preferred industries. The result is presented in a list format.",
      "Others": "Extra information about your investing strategy, maybe correlated with latest market and macroeconmic information and others. No more than 30 words."
   }}  
}}
{examples}
(END_OF_EXAMPLES)
Input data:
{input_data}
Macro data:
{macro_data}
Your investing style:
"""

INVESTING_STYLE_WITH_MACRO_DATA_EXAMPLE = """
Input data:
E/P,B/P,CF/P, S/P,Log-orthogonalized E/P,Log-orthogonalized B/P,Log-orthogonalized CF/P,Log-orthogonalized S/P,
Macro data:
The latest 1 year loan prime rate is 3.45. The latest month China CPI YOY growth rate is -0.5. The latest yield of China ten year government bonds is 2.6733%, while yield increases 0 BP over past one day, increases -4 BP over past one month, increases -21 BP over past half an year. The latest csi_300 pe is 10.9478, and the current PE ratio of the CSI 300 is at the 5.4 percentile over the past 5 years(0 indicates most undervalued, and 100 indicates most overvalued). The latest market sentiement index got 0.63 % return.
Your investing style:
{'Outline': 'A value-oriented investment approach focusing on fundamentally strong companies with a long-term perspective, leveraging current market undervaluation and stable economic indicators to build a diversified portfolio.',
 'Details': {'Risk Appetite': 'moderate',
  'Holding Period': 'more than one year',
  'Strategy Consistency': '0.85',
  'Rationality': '0.9',
  'StockPoolSelector': 'IndustryEqualStockSelector',
  'Others': 'Leverage low CPI and undervalued CSI 300 PE for potential upside.'}}
"""

INVSETING_STYLE_EXAMPLE = """
Example1:
Input data:
E/P,B/P,CF/P, S/P,Log-orthogonalized E/P,Log-orthogonalized B/P,Log-orthogonalized CF/P,Log-orthogonalized S/P,
Your investing style:
{{
   "Outline": "A value-driven investment approach focusing on stocks with strong fundamentals, undervalued valuations, and consistent cash flows over the long term.",  
   "Details": {{
      "Risk Appetite": "Moderately conservative",  
      "Holding Period": "More than one year",  
      "Strategy Consistency": "0.85",  
      "Rationality": "0.9",  
      "StockPoolSelector": {{  
         "IndustryBasisStockSelector": ["银行", "中药"] 
      }} 
   }}
}}

Example2:
Input data:
ROE, ROE_YEARLY, ROIC
Your investing style:
{{
   "Outline": "A value-driven investment approach focusing on stocks with strong fundamentals, undervalued valuations, and consistent cash flows over the long term.",  
   "Details": {{
      "Risk Appetite": "Moderately conservative",  
      "Holding Period": "More than one year",  
      "Strategy Consistency": "0.85",  
      "Rationality": "0.9",  
      "StockPoolSelector": "MVEqualStockSelector" 
   }}
}}
"""

INVESTING_DECISION_INSTRUCTION = """
Giving following 
1. input data in table format and their descriptions in json format.
2. investing style to make investment decisions in json format.
Please output {num_stocks} stocks you tend to invest. The res is in json format, key is "Stock", and value is a list containing stock code.Please make sure:
1. You output legal stock code. The stock code is legal if and only if it is in the input data "Stock" list.
2. The number of stock code is correct, actually equal to {num_stocks}.
Here is an example.
{examples}
(END OF EXAMPLES)
{input_data}
"""

INVESTING_DECISION_INSTRUCTION_USING_SELF_REFLECTION = """
Giving following 
1. input data in table format and decriptions in json format.
2. investing style to make investment decisions in json format.
3. investing decision history to make self reflections. You are accessible to history investing decision history, please make self reflections to make investment decisions.
Please output {num_stocks} stocks you tend to invest. The res is in json format, key is "Stock", and value is a list containing stock code. Please make sure:
1. You output legal stock code. The stock code is legal if and only if it is in the input data "Stock" list. The stock not in input data but in decision history is also illegal.
2. The number of stock code is correct, actually equal to {num_stocks}.
Here is an example. The example here does not use self reflection. However, in real cases, self reflection mechanism may be used. You should reflect and improve your decision process, then output the res.
{examples}
(END OF EXAMPLES)
{input_data}
(END OF INPUT data)
Here are investing decision history:
{decision_history}
"""

INVESTING_DECISION_EXAMPLE = """
For stock_nums in investing instructions, we use 3 in this example.
Input Data for investing decision:
1. Input Data Description:
{{ "E/P": "The inverse of the P/E ratio (E/P) indicates the earnings yield, showing the percentage of profit generated per dollar invested in the stock.",
"B/P": "Inverse of P/B (B/P) indicates the book yield, showing the return on book value per dollar invested.",
"S/P": "Inverse of P/S (S/P) reflects the sales yield, showing sales generated per dollar invested.",
"CF/P": "Inverse of P/CF (CF/P) shows the cash flow yield, representing cash flow generated per dollar invested.",
"Log-orthogonalized E/P": "Log-orthogonalized version of E/P, removing some kind of cap basis.",
"Log-orthogonalized B/P": ""Log-orthogonalized version of B/P, removing some kind of cap basis.",
"Log-orthogonalized CF/P": "Log-orthogonalized version of CF/P, removing some kind of cap basis.",
"Log-orthogonalized S/P": "Log-orthogonalized version of S/P, removing some kind of cap basis.",
"EBITDA/EV": "Measures a company's return on enterprise value, indicating operating earnings (EBITDA) generated per dollar of EV."
}}
2. Investing Style:
{{
   "Outline": "A value-driven investment approach focusing on stocks with strong fundamentals, undervalued valuations, and consistent cash flows over the long term.",  
   "Details": {{
      "Risk Appetite": "Moderately conservative",  
      "Holding Period": "More than one year",  
      "Strategy Consistency": "0.85",  
      "Rationality": "0.9",  
      "StockPoolSelector": "MVEqualStockSelector" 
   }}
}}
3. Input data:
',Stock,Date,E/P,B/P,CF/P,S/P,Log-orthogonalized E/P,Log-orthogonalized B/P,Log-orthogonalized CF/P,Log-orthogonalized S/P,EBITDA/EV\n965494,000858,20190102,0.06295366,0.30744636,0.038947526,0.19324197,-4.032941,-1.1295723,3.594055,-1.2754831,0.12488604\n2941460,002594,20190102,0.020888906,0.37708813,0.09185906,0.9017491,-4.038043,-0.6966869,5.084233,0.3152281,0.09258402\n7162558,600519,20190102,0.042301364,0.13605072,0.036664255,0.09038502,-7.6968794,-2.2439895,1.2049837,-2.2207088,0.079757534\n8104294,600900,20190102,0.066111766,0.40523565,0.1182603,0.15322393,-5.3881683,-1.0025798,3.743841,-1.5840118,0.105035394\n8267292,601012,20190102,0.062190603,0.30756927,0.032795224,0.41643697,-0.72993636,-0.7708632,5.801872,-0.31826368,0.088739015\n8431868,601288,20190102,0.16604953,1.2584949,0.12149128,0.4757528,-7.5973797,-0.1158539,1.556502,-0.6272717,0.05945474\n8665067,601888,20190102,0.02850359,0.1358013,0.034710173,0.35662726,-3.433404,-1.7193639,4.34933,-0.59344673,0.051195413\n9068489,603259,20190102,0.024971908,0.12955885,0.018961666,0.10751114,-2.9358995,-1.8100101,4.314365,-1.7471998,0.04303389\n'

LLM output:
{{'Stock': ['000858', '600900', '601288']}}

Note that in this example, we ask LLM to output 3 stocks. However, in real scenarios, you should follow the "num_stocks" args in the instruction.
"""












SUMMARIZE_INSTRUCTION = """
请用中文总结以下网页内容，要求总结简洁明了，不超过70个字，去除所有冗余文字，确保信息密度最大化。直接输出核心信息，无需额外解释或修饰。
你的输出格式应为: 对于新闻提到的每个上市公司主体 / 金融市场大盘, 及其简要地概括发生的具体事件，省略其他所有的冗余的,修饰性表述，只给出新闻主干。
Here is an example.
{example}
Input data
{input_data}
Res
"""

SUMMARIZE_EXAMPLE = """
Input data
'<HTML>  <DIV>    <DIV>      <P>01月01日消息，亿邦动力网第一时间了解到，<AI id='08R9CCC31E' type='2' parm='{}'>腾讯</AI></P>      <P nodelevel="1">· 12月18日上午，<AI id='08R9CCC31E' type='2' parm='{}'>腾讯</AI>和<AI id='7jAIgmkKPQ' type='2' parm='{}'>京东</AI>宣布入股<AI id='ivrPzhsRuc' type='2' parm='{}'>唯品会</AI>，分别以现金6.04亿美元和2.59亿美元认购<AI id='ivrPzhsRuc' type='2' parm='{}'>唯品会</AI>的股票。算上<AI id='7jAIgmkKPQ' type='2' parm='{}'>京东</AI>已经持有的2.5%，交易完成后，<AI id='08R9CCC31E' type='2' parm='{}'>腾讯</AI>和<AI id='7jAIgmkKPQ' type='2' parm='{}'>京东</AI>将分别拥有<AI id='ivrPzhsRuc' type='2' parm='{}'>唯品会</AI>7% 和5.5% 的股票。<AI id='08R9CCC31E' type='2' parm='{}'>腾讯</AI>已经是第二大股东，仅次于<AI id='ivrPzhsRuc' type='2' parm='{}'>唯品会</AI>创始人<AI id='198394' type='1' parm='{}'>沈亚</AI>。</P>      <P nodelevel="1">本文由亿邦智能机器人ebrunGo撰写，机器人还很年轻，如有纰漏，还望指正。</P>    </DIV>  </DIV></HTML><!--<AI id='08R9CCC31E' type='2' parm='{}'>腾讯</AI>：<AI id='601933.SH' type='6' parm='{}'>永辉超市</AI>市值目前846亿-->'
Res
腾讯和京东入股唯品会，分别持股7%和5.5%，腾讯成为第二大股东。唯品会创始人持股比例降至第二。
"""
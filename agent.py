from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from tools import (
    get_stock_data,
    get_stock_info,
    get_company_info,
    get_stock_dividends,
    get_income_stmt,
    get_balance_sheet,
    get_cash_flow,
    get_analyst_recommendations,
)
import prompts
from llms import get_together_lm

# Define tools for each function
get_stock_data_tool = FunctionTool.from_defaults(
    fn=get_stock_data,
    name="stock_data",
    description="Fetches stock historical data which is  a comprehensive record of a company's stock price movements over time. This data provides a detailed view of the stock's performance and includes various key metrics such as open price, close price, high price, low price, volume and adjusted close price",
)
get_stock_info_tool = FunctionTool.from_defaults(
    fn=get_stock_info,
    name="stock_information",
    description="Stock information from yahoo finance includes key data such as current price, market cap, volume, financial ratios, dividend yield, and analyst ratings, along with historical price and volume data. This comprehensive snapshot helps investors assess the stock's performance and make informed decisions.",
)
get_company_info_tool = FunctionTool.from_defaults(
    fn=get_company_info,
    name="company_information",
    description="Company information available in yahoo finance includes details such as market capitalization, sector, industry, full and short names, website, and financial metrics like revenue, net income, and total debt. This data provides a thorough overview of the company's financial health and market position.",
)
get_stock_dividends_tool = FunctionTool.from_defaults(
    fn=get_stock_dividends,
    name="stock_dividend",
    description="The stock_dividends data in yahoo finance includes information on dividend payments such as dividend dates, amounts, and yields. It provides details on past and upcoming dividends, helping investors assess the income potential from their investments.",
)
get_income_stmt_tool = FunctionTool.from_defaults(
    fn=get_income_stmt,
    name="income_statements",
    description="The income_statements data in yahoo finance includes detailed financial performance metrics such as revenue, gross profit, operating expenses, net income, and earnings per share (EPS). This data helps investors evaluate a company's profitability and financial health over specific reporting periods.",
)
get_balance_sheet_tool = FunctionTool.from_defaults(
    fn=get_balance_sheet,
    name="balance_sheet",
    description="The balance sheet data in yahoo finance includes detailed information on a company's assets, liabilities, and shareholders' equity. This data provides a snapshot of the company's financial position, helping investors assess its stability and financial health.",
)
get_cash_flow_tool = FunctionTool.from_defaults(
    fn=get_cash_flow,
    name="cash_flow",
    description="The cash flow statements in yahoo finance provide detailed information on a company's cash inflows and outflows from operating, investing, and financing activities. This data helps investors understand the company's liquidity, cash management, and overall financial flexibility.",
)
get_analyst_recommendations_tool = FunctionTool.from_defaults(
    fn=get_analyst_recommendations,
    name="analyst_recommendation",
    description="Displays sell and buy decisions made by a set of trained and experienced analyst",
)


prefix_msgs = [
    ChatMessage(role=MessageRole.SYSTEM, content=prompts.GPT_FINANCIAL_ANALYST_SYS_STR),
    ChatMessage(
        role=MessageRole.USER,
        content="Use emojis in the response and Before response, I want you to think through is that the response generated is correct and elaborate on your reasoning before you give the final answer.",
    ),
]


financial_react_agent = ReActAgent.from_tools(
    [
        get_stock_data_tool,
        get_stock_info_tool,
        get_company_info_tool,
        get_stock_dividends_tool,
        get_income_stmt_tool,
        get_balance_sheet_tool,
        get_cash_flow_tool,
        get_analyst_recommendations_tool,
    ],
    llm=get_together_lm(),
    prefix_messages=prefix_msgs,
    token_counting=True,
    verbose=True,
    max_iterations=20,
)

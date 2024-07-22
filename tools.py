from llama_index.core import PromptTemplate
import yfinance as yf
from pandas import DataFrame
from typing import Annotated, Callable, Any, Optional


from llms import get_groq_lm


llm = get_groq_lm()


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    # Fetch stock data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(start=start_date, end=end_date).describe()
    return stock_data.to_string()


def get_balance_sheet(symbol: str) -> str:
    # Fetch balance sheet from Yahoo Finance
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet

    # Prompt template for summarizing financial data
    template_financial_summary = """You are a financial analyst.
    Provide a detailed summary of the key insights from the following balance sheet,
    ensuring no essential data is omitted. Keep the summary within 50 words.\n\n{balance_sheet}"""

    prompt_balance_sheet = PromptTemplate(template=template_financial_summary)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(
        prompt_balance_sheet.format(balance_sheet=balance_sheet)
    )

    return result_summary.text


def get_stock_info(symbol: str) -> str:
    # Fetch stock information from Yahoo Finance
    ticker = yf.Ticker(symbol)
    info = ticker.info

    stock_info = str(info)

    # Prompt template for summarizing financial data
    template_stock_prices = """You are a financial analyst.
    Provide a detailed summary of the key insights from the following stock information.
    Ensure the summary is comprehensive and fits within 3 lines.\n\n{stock_info}"""

    prompt_stock_proces = PromptTemplate(template=template_stock_prices)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_stock_proces.format(stock_info=stock_info))

    return result_summary.text


def get_company_info(
    symbol: str,
) -> str:
    # Fetch company information from Yahoo Finance
    ticker = yf.Ticker(symbol)
    info = ticker.info
    company_info = {
        "Company Name": info.get("shortName", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Country": info.get("country", "N/A"),
        "Website": info.get("website", "N/A"),
    }
    company_info_df = DataFrame([company_info])
    company_info = str(company_info_df)

    # Prompt template for summarizing financial data
    template_company_info = """You are a financial analyst.
    Provide a detailed summary of the key insights from the following company information,
    ensuring no essential data is omitted.\n\n{company_info}"""

    prompt_company_info = PromptTemplate(template=template_company_info)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_company_info.format(company_info=company_info))

    return result_summary.text


def get_income_stmt(symbol: str) -> str:
    # Fetch income statement from Yahoo Finance
    ticker = yf.Ticker(symbol)
    income_stmt = ticker.financials

    income_stmt = income_stmt.to_string()

    # Prompt template for summarizing financial data
    template_income_stmt = """You are a financial analyst.
    Provide a detailed summary of the key insights from the following income statement,
    ensuring no essential data is omitted. Keep the summary within 3 lines.\n\n{income_stmt}"""

    prompt_income_stmt = PromptTemplate(template=template_income_stmt)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_income_stmt.format(income_stmt=income_stmt))

    return result_summary.text


def get_cash_flow(symbol: str) -> str:
    # Fetch cash flow from Yahoo Finance
    ticker = yf.Ticker(symbol)
    cash_flows = ticker.cashflow
    cash_flows = cash_flows.to_string()

    # Prompt template for summarizing financial data
    template_cash_flows_template = """You are a financial analyst.
    Provide a detailed summary of the key insights from the following cash flow data,
    ensuring no essential data is omitted.\n\n{cash_flows}"""

    prompt_cash_flows = PromptTemplate(template=template_cash_flows_template)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(prompt_cash_flows.format(cash_flows=cash_flows))

    return result_summary.text


def get_stock_dividends(symbol: str) -> str:
    # Fetch stock dividends from Yahoo Finance
    ticker = yf.Ticker(symbol)
    dividends = ticker.dividends

    dividends_info = dividends.to_string()

    # Prompt template for summarizing financial data
    template_dividends_info = """You are a financial analyst.
    Provide a detailed summary of the key trends from the following dividend data,
    ensuring no essential data is omitted. Keep the summary concise.\n\n{dividends_info}"""

    prompt_dividends_info = PromptTemplate(template=template_dividends_info)

    # Use Llama Index with chosen LLM model
    result_summary = llm.complete(
        prompt_dividends_info.format(dividends_info=dividends_info)
    )

    return result_summary.text


def get_analyst_recommendations(symbol: str) -> str:
    # Fetch analyst recommendations from Yahoo Finance
    ticker = yf.Ticker(symbol)
    recommendations = ticker.recommendations
    if recommendations.empty:
        return "No recommendations available"

    recommendation_counts = recommendations[
        ["strongBuy", "buy", "hold", "sell", "strongSell"]
    ].sum()
    most_common_recommendation = recommendation_counts.idxmax()
    count = recommendation_counts.max()

    return f"Most common recommendation: {most_common_recommendation} ({count} votes)"

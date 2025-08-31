import os
import warnings
import math
import time
from dotenv import load_dotenv
import pytz
from datetime import datetime
import yfinance as yf
import pandas as pd
import feedparser  # 🔹 for news fetching
# Optional SerpAPI name discovery
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False
# suppress noisy future warnings from yfinance
warnings.simplefilter(action="ignore", category=FutureWarning)
# load .env for SERPAPI (optional)
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# Import news from test_news_analyst
try:
    from tests.test_news_analyst import fetch_news
    print("Using news analyst from tests")
except ImportError:
    try:
        from agent.test_news_analyst import fetch_news
        print("Using news analyst from agent")
    except ImportError:
        print("News analyst not available")
        def fetch_news(query, limit=5, days_limit=60):
            return []

# -------------------------
# Helpers
# -------------------------
def normalize_name(text: str) -> str:
    """Return uppercase alphanumeric-only version for candidate ticker construction."""
    return "".join(ch for ch in text if ch.isalnum()).upper()

# A small manual mapping for common names
NAME_TO_TICKER = {
    "RELIANCE": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "TITAN": "TITAN.NS",
    "BEL": "BEL.NS",
    "BHEL": "BHEL.NS",
    "TCS": "TCS.NS",
    "CESC": "CESC.NS",  # Added CESC mapping
    # extend as needed
}

def check_ticker_exists(ticker: str) -> bool:
    """Quick check if a ticker returns plausible data from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return False
        return bool(
            info.get("currentPrice") is not None
            or info.get("regularMarketPrice") is not None
            or info.get("shortName")
        )
    except Exception:
        return False

def serpapi_discover_symbol(query: str) -> str | None:
    """Try to discover symbol using SerpAPI Google Finance (optional)."""
    if not SERPAPI_KEY or not SERPAPI_AVAILABLE:
        return None
    try:
        params = {"engine": "google_finance", "q": query, "api_key": SERPAPI_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()
        finance_res = results.get("finance", {}).get("result", [])
        if finance_res and isinstance(finance_res, list):
            sym = finance_res[0].get("symbol")
            if sym:
                return sym
    except Exception:
        pass
    return None

def resolve_ticker_from_input(user_input: str) -> str:
    """
    Resolve user input to a ticker:
      - If input already includes '.' assume it's a ticker
      - Try NAME_TO_TICKER mapping
      - Try normalized + .NS
      - Try normalized without suffix
      - Try SerpAPI
      - Fallback: normalized + .NS
    """
    s = user_input.strip()
    if not s:
        raise ValueError("Empty input")
    if "." in s:
        return s.upper()
    key = s.strip().upper()
    if key in NAME_TO_TICKER:
        return NAME_TO_TICKER[key]
    norm = normalize_name(s)
    if not norm:
        raise ValueError("Could not normalize input to candidate ticker")
    candidate_ns = norm + ".NS"
    if check_ticker_exists(candidate_ns):
        return candidate_ns
    if check_ticker_exists(norm):
        return norm
    discovered = serpapi_discover_symbol(s)
    if discovered and check_ticker_exists(discovered):
        return discovered
    return candidate_ns

# -------------------------
# Financial indicators
# -------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> float | None:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_valid = rsi.dropna()
    if rsi_valid.empty:
        return None
    return float(rsi_valid.iloc[-1])

def calculate_pivot(high: float, low: float, close: float) -> dict:
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {"Pivot": pivot, "R1": r1, "S1": s1, "R2": r2, "S2": s2}

def human_readable_number(n):
    try:
        n = float(n)
    except Exception:
        return n
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

def interpret_rsi(rsi: float | None) -> str:
    if rsi is None:
        return "N/A"
    if rsi < 30:
        return f"{rsi:.2f} 🟢 (Oversold)"
    if rsi > 70:
        return f"{rsi:.2f} 🔴 (Overbought)"
    return f"{rsi:.2f} 🔵 (Neutral)"

def interpret_ma(ma20: float | None, ma50: float | None) -> str | None:
    if ma20 is None or ma50 is None:
        return None
    if ma20 > ma50:
        return f"MA20 {ma20:.2f} > MA50 {ma50:.2f} 🟢 (Bullish)"
    if ma20 < ma50:
        return f"MA20 {ma20:.2f} < MA50 {ma50:.2f} 🔴 (Bearish)"
    return f"MA20 {ma20:.2f} = MA50 {ma50:.2f} 🔵 (Neutral)"

def combined_summary(rsi: float | None, ma20: float | None, ma50: float | None) -> str:
    bulls = bears = 0
    if rsi is not None:
        if rsi < 30:
            bulls += 1
        elif rsi > 70:
            bears += 1
    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            bulls += 1
        elif ma20 < ma50:
            bears += 1
    if bulls > 0 and bears == 0:
        return "📈 Bullish"
    if bears > 0 and bulls == 0:
        return "📉 Bearish"
    if bulls == 0 and bears == 0:
        return "⚖️ Neutral"
    return "⚖️ Mixed"

# -------------------------
# Yahoo Finance corporate actions (dividends & splits)
# -------------------------
def fetch_corporate_actions_yahoo(ticker: str, limit: int = 5) -> list[dict]:
    """
    Return recent corporate actions from Yahoo Finance:
    - Dividend amounts
    - Stock split ratios
    Each item: {"date": "YYYY-MM-DD", "type": "Dividend|Split", "value": <float|str>}
    """
    try:
        t = yf.Ticker(ticker)
        out: list[dict] = []
        # Prefer combined dataframe if available
        actions_df = getattr(t, "actions", pd.DataFrame())
        if isinstance(actions_df, pd.DataFrame) and not actions_df.empty:
            # Keep only recent rows
            actions_df = actions_df.tail(limit * 2)  # little extra to cover both types
            for idx, row in actions_df.iterrows():
                date_str = str(getattr(idx, "date", lambda: idx)())
                # Dividends
                div_val = row.get("Dividends")
                if div_val is not None and not pd.isna(div_val) and float(div_val) != 0.0:
                    out.append({"date": date_str, "type": "Dividend", "value": float(div_val)})
                # Splits
                split_val = row.get("Stock Splits")
                if split_val is not None and not pd.isna(split_val) and float(split_val) != 0.0:
                    out.append({"date": date_str, "type": "Split", "value": float(split_val)})
        # Fallback to individual series if needed
        if not out:
            div = t.dividends
            if isinstance(div, pd.Series) and not div.empty:
                for d, v in div.tail(limit).items():
                    out.append({"date": str(d.date()), "type": "Dividend", "value": float(v)})
            splits = t.splits
            if isinstance(splits, pd.Series) and not splits.empty:
                for d, v in splits.tail(limit).items():
                    out.append({"date": str(d.date()), "type": "Split", "value": float(v)})
        # Sort newest first and cap to limit
        out.sort(key=lambda x: x["date"], reverse=True)
        return out[:limit]
    except Exception:
        return []

def format_split_ratio(ratio: float) -> str:
    """
    Convert Yahoo split ratio (e.g., 2.0, 0.5) to human string like '2-for-1' or '1-for-2'.
    """
    try:
        r = float(ratio)
        if r >= 1:
            return f"{int(round(r))}-for-1"
        else:
            inv = int(round(1 / r)) if r != 0 else 0
            return f"1-for-{inv}"
    except Exception:
        return str(ratio)

# -------------------------
# Main fetch & chat
# -------------------------
def download_with_retry(ticker, period="12mo", interval="1d", max_retries=3, retry_delay=2):
    """Download stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            if not df.empty:
                return df
            print(f"Attempt {attempt + 1}: Empty data for {ticker}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error downloading {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

def fetch_stock_data(user_input: str) -> dict:
    try:
        ticker = resolve_ticker_from_input(user_input)
    except Exception as e:
        return {"error": f"Ticker resolution failed: {e}"}
    
    try:
        # Download data with retry logic
        df = download_with_retry(ticker)
        if df.empty:
            return {"error": f"No historical data found for {ticker} after multiple attempts"}
        
        last = df.iloc[-1]
        price = float(last["Close"])
        vol = int(last["Volume"])
        rsi = calculate_rsi(df["Close"])
        pivots = calculate_pivot(float(last["High"]), float(last["Low"]), float(last["Close"]))
        ma20 = float(df["Close"].rolling(window=20).mean().iloc[-1]) if len(df) >= 20 else None
        ma50 = float(df["Close"].rolling(window=50).mean().iloc[-1]) if len(df) >= 50 else None
        
        # Get ticker info with retry logic
        tkr = None
        for attempt in range(3):
            try:
                tkr = yf.Ticker(ticker)
                ticker_info = tkr.info
                if ticker_info and "shortName" in ticker_info:
                    break
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error getting ticker info for {ticker}: {e}")
                if attempt < 2:
                    time.sleep(1)
        
        if not tkr or not ticker_info:
            return {"error": f"Could not get ticker info for {ticker}"}
        
        market_cap = ticker_info.get("marketCap")
        pe = ticker_info.get("trailingPE")
        fifty2_high = ticker_info.get("fiftyTwoWeekHigh")
        fifty2_low = ticker_info.get("fiftyTwoWeekLow")
        prev_close = ticker_info.get("previousClose")
        exchange = tkr.fast_info.get("exchange", "N/A")
        
        # Get news using the news agent
        news = fetch_news(ticker, limit=5, days_limit=120)
        
        return {
            "ticker": ticker,
            "price": price,
            "currency": ticker_info.get("currency", "INR"),
            "exchange": exchange,
            "volume": vol,
            "rsi": rsi,
            "pivots": pivots,
            "ma20": ma20,
            "ma50": ma50,
            "market_cap": market_cap,
            "pe_ratio": pe,
            "52w_high": fifty2_high,
            "52w_low": fifty2_low,
            "prev_close": prev_close,
            "summary": combined_summary(rsi, ma20, ma50),
            "news": news,
            "corporate_actions": fetch_corporate_actions_yahoo(ticker, limit=5),
        }
    except Exception as e:
        return {"error": str(e)}

def chat():
    print("\n💬 Stock Chat — type company name or ticker (e.g. 'TATA STEEL' or 'TITAN') — type 'exit' to quit\n")
    while True:
        ui = input("You: ").strip()
        if ui.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break
        res = fetch_stock_data(ui)
        if "error" in res:
            print(f"⚠️ Error: {res['error']}\n")
            continue
        print(f"\n📊 {res['ticker']}")
        print(f"   • Price: {res['price']:.2f} {res['currency']}")
        print(f"   • Exchange: {res['exchange']}")
        print(f"   • Volume: {human_readable_number(res['volume'])}")
        print(f"   • RSI: {interpret_rsi(res['rsi'])}")
        print(f"   • Market Cap: {human_readable_number(res['market_cap'])}")
        print(f"   • P/E Ratio: {res['pe_ratio']}")
        print(f"   • 52W High / Low: {res['52w_high']} / {res['52w_low']}")
        print(f"   • Previous Close: {res['prev_close']}")
        if res['ma20'] is not None:
            print(f"   • MA20: {res['ma20']:.2f}")
        if res['ma50'] is not None:
            print(f"   • MA50: {res['ma50']:.2f}")
        ma_sig = interpret_ma(res['ma20'], res['ma50'])
        if ma_sig:
            print(f"   • {ma_sig}")
        print(f"\nSummary Signal: {res['summary']}")
        # Show news
        if res.get("news"):
            print("\n📰 Recent News:")
            for item in res["news"]:
                print(f"   • {item['published']} – {item['title']}")
                print(f"     🔗 {item['link']}")
        # Show corporate actions (Yahoo)
        if res.get("corporate_actions"):
            print("\n🏢 Corporate Actions (Yahoo):")
            for ca in res["corporate_actions"]:
                if ca["type"] == "Dividend":
                    print(f"   • {ca['date']} — Dividend: {ca['value']}")
                elif ca["type"] == "Split":
                    print(f"   • {ca['date']} — Split: {format_split_ratio(ca['value'])}")
                else:
                    print(f"   • {ca['date']} — {ca['type']}: {ca['value']}")
        print("")

if __name__ == "__main__":
    chat()
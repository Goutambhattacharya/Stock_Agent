from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import logging
import pandas as pd
import io
import json
from datetime import datetime
import yfinance as yf
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Try to import agents with specific priority to agent folder
def import_agents():
    """Try to import agents with priority to agent folder"""
    agents = {
        'market_data': None,
        'news': None
    }
    
    # Try importing market data agent with priority to agent folder
    market_data_imports = [
        'agent.test_market_data_agent',  # Prioritize agent folder
        'test_market_data_agent',
        'tests.test_market_data_agent',
        'agents.test_market_data_agent'
    ]
    
    for module_name in market_data_imports:
        try:
            module = __import__(module_name, fromlist=['fetch_stock_data', 'resolve_ticker_from_input', 'format_split_ratio'])
            agents['market_data'] = {
                'fetch_stock_data': getattr(module, 'fetch_stock_data'),
                'resolve_ticker_from_input': getattr(module, 'resolve_ticker_from_input'),
                'format_split_ratio': getattr(module, 'format_split_ratio')
            }
            logger.info(f"Successfully imported market data agent from {module_name}")
            break
        except ImportError as e:
            logger.debug(f"Failed to import market data agent from {module_name}: {e}")
    
    # Try importing news agent with priority to agent folder
    news_imports = [
        'agent.test_news_analyst',  # Prioritize agent folder
        'test_news_analyst',
        'tests.test_news_analyst',
        'agents.test_news_analyst'
    ]
    
    for module_name in news_imports:
        try:
            module = __import__(module_name, fromlist=['fetch_news'])
            agents['news'] = {
                'fetch_news': getattr(module, 'fetch_news')
            }
            logger.info(f"Successfully imported news analyst from {module_name}")
            # Log the module file path to confirm we're using the right one
            if hasattr(module, '__file__'):
                logger.info(f"News analyst module path: {module.__file__}")
            break
        except ImportError as e:
            logger.debug(f"Failed to import news analyst from {module_name}: {e}")
    
    return agents

# Import agents
agents = import_agents()

# Create fallback functions if agents aren't available
if not agents['market_data']:
    logger.warning("Market data agent not available. Using fallback functions.")
    
    def resolve_ticker_from_input(user_input: str) -> str:
        return user_input.strip().upper()
    
    def format_split_ratio(ratio: float) -> str:
        try:
            r = float(ratio)
            if r >= 1:
                return f"{int(round(r))}-for-1"
            else:
                inv = int(round(1 / r)) if r != 0 else 0
                return f"1-for-{inv}"
        except Exception:
            return str(ratio)
    
    def download_with_retry(ticker, period="12mo", interval="1d", max_retries=3, retry_delay=2):
        """Download stock data with retry logic"""
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
                if not df.empty:
                    return df
                logger.info(f"Attempt {attempt + 1}: Empty data for {ticker}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logger.info(f"Attempt {attempt + 1}: Error downloading {ticker}: {e}")
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
            
            # Calculate simple RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
            
            # Calculate moving averages
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
                    logger.info(f"Attempt {attempt + 1}: Error getting ticker info for {ticker}: {e}")
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
            if agents['news']:
                news = agents['news']['fetch_news'](ticker, limit=5, days_limit=120)
            else:
                news = []
            
            return {
                "ticker": ticker,
                "price": price,
                "currency": ticker_info.get("currency", "INR"),
                "exchange": exchange,
                "volume": vol,
                "rsi": rsi_value,
                "ma20": ma20,
                "ma50": ma50,
                "market_cap": market_cap,
                "pe_ratio": pe,
                "52w_high": fifty2_high,
                "52w_low": fifty2_low,
                "prev_close": prev_close,
                "summary": "Neutral",  # Simplified
                "news": news,
                "corporate_actions": [],  # Simplified
            }
        except Exception as e:
            return {"error": str(e)}
    
    agents['market_data'] = {
        'fetch_stock_data': fetch_stock_data,
        'resolve_ticker_from_input': resolve_ticker_from_input,
        'format_split_ratio': format_split_ratio
    }
12
if not agents['news']:
    logger.warning("News analyst not available. Using fallback function.")
    
    def fetch_news(query, limit=5, days_limit=120):
        return [{"title": "News analyst not available", "link": "", "published": ""}]
    
    agents['news'] = {
        'fetch_news': fetch_news
    }

# Helper function to transform agent response to template format
def transform_agent_response(ticker, agent_response):
    """Transform agent response to match template expectations"""
    # Create a results dictionary with defaults
    results = {
        'ticker': ticker,
        'summary': 'Neutral',
        'price': 0.0,
        'currency': 'USD',
        'exchange': 'NASDAQ',
        'volume': 0,
        'rsi': 0.0,
        'ma20': 0.0,
        'ma50': 0.0,
        'market_cap': 0,
        'pe_ratio': 0.0,
        '52w_high': 0.0,
        '52w_low': 0.0,
        'historical': None,  # Set to None to avoid displaying the table
        'historical_columns': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],  # Keep columns for template compatibility
        'news': [],
        'corporate_actions': [],
        'debug_info': {},  # Add empty debug_info to avoid template error
        'raw_output': str(agent_response)
    }
    
    # Update with agent response if available
    if agent_response and 'error' not in agent_response:
        # Basic info
        results['ticker'] = agent_response.get('ticker', ticker)
        results['price'] = agent_response.get('price', 0.0)
        results['currency'] = agent_response.get('currency', 'USD')
        results['exchange'] = agent_response.get('exchange', 'NASDAQ')
        results['volume'] = agent_response.get('volume', 0)
        results['market_cap'] = agent_response.get('market_cap', 0)
        
        # Technical indicators
        results['rsi'] = agent_response.get('rsi', 50.0)
        results['ma20'] = agent_response.get('ma20', results['price'] * 0.99)
        results['ma50'] = agent_response.get('ma50', results['price'] * 0.98)
        results['pe_ratio'] = agent_response.get('pe_ratio', 25.0)
        results['52w_high'] = agent_response.get('52w_high', results['price'] * 1.2)
        results['52w_low'] = agent_response.get('52w_low', results['price'] * 0.8)
        
        # Summary
        results['summary'] = agent_response.get('summary', 'Neutral')
        
        # News - already formatted by the news agent
        if 'news' in agent_response:
            results['news'] = agent_response['news']
        
        # Corporate actions - format split ratios
        if 'corporate_actions' in agent_response:
            formatted_actions = []
            for action in agent_response['corporate_actions']:
                formatted_action = action.copy()
                if action['type'] == 'Split':
                    formatted_action['value'] = agents['market_data']['format_split_ratio'](action['value'])
                formatted_actions.append(formatted_action)
            results['corporate_actions'] = formatted_actions
    
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle both GET and POST requests for the main page"""
    if request.method == 'POST':
        # Handle form submission
        ticker_input = request.form.get('ticker', '').strip()
        if not ticker_input:
            return render_template('index.html', results={'error': 'Please enter a stock ticker or company name'})
        
        try:
            logger.info(f"Analyzing stock: {ticker_input}")
            
            # Get analysis from market data agent
            agent_response = agents['market_data']['fetch_stock_data'](ticker_input)
            
            # Transform agent response to match template format
            results = transform_agent_response(ticker_input, agent_response)
            
            return render_template('index.html', results=results)
        except Exception as e:
            logger.error(f"Error processing ticker {ticker_input}: {e}")
            return render_template('index.html', results={'error': str(e)})
    else:
        # GET request: just render the form
        return render_template('index.html')

@app.route('/download/<ticker>')
def download_csv(ticker):
    """Download historical data as CSV"""
    try:
        ticker = ticker.strip().upper()
        
        # Get historical data with retry logic
        def download_with_retry(ticker, period="12mo", interval="1d", max_retries=3, retry_delay=2):
            for attempt in range(max_retries):
                try:
                    df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
                    if not df.empty:
                        return df
                    logger.info(f"Attempt {attempt + 1}: Empty data for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except Exception as e:
                    logger.info(f"Attempt {attempt + 1}: Error downloading {ticker}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            return pd.DataFrame()  # Return empty DataFrame if all retries fail
        
        hist_df = download_with_retry(ticker)
        
        if hist_df.empty:
            return "No historical data available", 404
        
        # Reset index to make Date a column
        hist_df = hist_df.reset_index()
        
        # Create CSV in memory
        output = io.StringIO()
        hist_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{ticker}_historical_data.csv'
        )
    except Exception as e:
        logger.error(f"Error downloading CSV for {ticker}: {e}")
        return f"Error downloading data: {str(e)}", 500

@app.route('/api/stock/<symbol>')
def api_stock(symbol):
    """API endpoint to get stock data"""
    try:
        symbol = symbol.strip().upper()
        logger.info(f"API request for stock: {symbol}")
        
        stock_data = agents['market_data']['fetch_stock_data'](symbol)
        return jsonify(stock_data)
    except Exception as e:
        logger.error(f"API error for stock {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/news/<symbol>')
def api_news(symbol):
    """API endpoint to get news data"""
    try:
        symbol = symbol.strip().upper()
        logger.info(f"API request for news: {symbol}")
        
        news_data = agents['news']['fetch_news'](symbol)
        return jsonify(news_data)
    except Exception as e:
        logger.error(f"API error for news {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'agents': {
            'market_data_agent': 'available' if agents['market_data'] else 'unavailable',
            'news_analyst': 'available' if agents['news'] else 'unavailable'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

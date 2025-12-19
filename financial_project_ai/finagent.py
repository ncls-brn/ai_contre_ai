import gradio as gr
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import re
import os
from dotenv import load_dotenv
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates


# Charger les variables d'environnement
load_dotenv()

def load_api_key():
    """Load API key from environment variable"""
    apikey = os.getenv('MISTRAL_API_KEY')
    
    if not apikey:
        raise ValueError(
            "❌ MISTRAL_API_KEY not found!\n"
            "Create a .env file with:\n"
            "MISTRAL_API_KEY=your_actual_key_here"
        )
    
    return apikey

class TickerExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def extract_from_text(self, text):
        """Extract ticker from text using regex patterns"""
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|equity))',
            r'\b([A-Z]{1,5})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                for potential_ticker in matches:
                    try:
                        ticker = yf.Ticker(potential_ticker)
                        info = ticker.info
                        if info:
                            return potential_ticker
                    except:
                        continue
        return None

    def get_ticker_from_ai(self, text):
        """Use Mistral AI to identify potential stock ticker from text"""
        prompt = f"Extract ONLY the stock ticker symbol from: {text}. Return only the ticker, nothing else."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 10
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                potential_ticker = result['choices'][0]['message']['content'].strip().upper()
                potential_ticker = re.sub(r'[^A-Z]', '', potential_ticker)
                
                try:
                    ticker = yf.Ticker(potential_ticker)
                    info = ticker.info
                    if info and len(potential_ticker) <= 5:
                        return potential_ticker
                except:
                    return None
            return None
        except Exception as e:
            print(f"AI extraction error: {str(e)}")
            return None

class FinancialAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.ticker_extractor = TickerExtractor(api_key)
        self.context = "You are a financial expert. Analyze technical indicators and provide clear, actionable insights."
        self.conversation_history = []

    def get_stock_data(self, ticker, period="1y"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                return None
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    def create_technical_analysis(self, df):
        """Add technical indicators to dataframe"""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Close'])
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['ATR'] = self.calculate_atr(df)
        df['Bollinger_Upper'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        return df

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, lower_band

    def analyze_technical_signals(self, df):
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'rsi_signal': 'oversold' if latest['RSI'] < 30 else 'overbought' if latest['RSI'] > 70 else 'neutral',
            'macd_signal': 'bullish' if latest['MACD'] > latest['Signal'] and prev['MACD'] <= prev['Signal'] 
                          else 'bearish' if latest['MACD'] < latest['Signal'] and prev['MACD'] >= prev['Signal'] 
                          else 'neutral',
            'ma_signal': 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish',
        }
        
        signals['price_to_sma20'] = (latest['Close'] / latest['SMA_20'] - 1) * 100
        return signals

    def generate_market_insights(self, df, ticker):
        signals = self.analyze_technical_signals(df)
        returns = df['Daily_Return'].iloc[-20:].mean() * 100
        volatility = df['Volatility'].iloc[-1] * 100
        
        insights = f"""Analysis for {ticker}:

Technical Indicators:
- RSI: {df['RSI'].iloc[-1]:.2f} ({signals['rsi_signal']})
- MACD: {signals['macd_signal']} momentum
- Price vs SMA20: {signals['price_to_sma20']:.2f}%
- MA Signal: {signals['ma_signal']}

Risk Metrics:
- 20-day Volatility: {volatility:.2f}%
- Avg Daily Return: {returns:.2f}%"""
        return insights

    def get_response(self, message, include_analysis=True):
        if include_analysis and hasattr(self, 'df'):
            insights = self.generate_market_insights(self.df, self.selected_ticker)
            message += f"\n\n{insights}"
        
        messages = [{"role": "system", "content": self.context}]
        for msg in self.conversation_history:
            messages.append(msg)
        messages.append({"role": "user", "content": message})
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

class FinancialPlotter:
    @staticmethod
    def create_stock_plot(df, ticker):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz:
            df.index = df.index.tz_localize(None)
        df['Date_num'] = mpdates.date2num(df.index)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ohlc_data = df[['Date_num', 'Open', 'High', 'Low', 'Close']].values
        candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='green', colordown='red', alpha=0.8)
        ax.plot(df['Date_num'], df['SMA_20'], color='orange', label='SMA 20', linewidth=1.5)
        ax.plot(df['Date_num'], df['SMA_50'], color='blue', label='SMA 50', linewidth=1.5)
        ax.set_title(f'{ticker} Stock Price', fontsize=14)
        ax.set_ylabel('Price')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mpdates.ConciseDateFormatter(mpdates.AutoDateLocator()))
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_technical_indicators_plot(df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax1.set_title('RSI (14)')
        ax1.set_ylabel('RSI Value')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
        ax1.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
        
        ax2.plot(df.index, df['MACD'], color='blue', label='MACD', linewidth=1.5)
        ax2.plot(df.index, df['Signal'], color='orange', label='Signal', linewidth=1.5)
        ax2.bar(df.index, df['MACD'] - df['Signal'], 
                color=['g' if val >= 0 else 'r' for val in df['MACD'] - df['Signal']],
                alpha=0.3)
        ax2.set_title('MACD')
        ax2.set_ylabel('Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_volume_analysis_plot(df):
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['g' if c >= o else 'r' for c, o in zip(df['Close'], df['Open'])]
        ax.bar(df.index, df['Volume'], color=colors, alpha=0.7)
        volume_ma = df['Volume'].rolling(window=20).mean()
        ax.plot(df.index, volume_ma, color='blue', label='Volume MA (20)', linewidth=1.5)
        ax.set_title('Volume Analysis')
        ax.set_ylabel('Volume')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        return fig

def clear_outputs():
    return [], None, None, None

def process_request(message, ticker_input, period_input, history):
    if history is None:
        history = []
    
    if not message or not message.strip():
        return history, None, None, None
    
    try:
        apikey = load_api_key()
    except Exception as e:
        error_msg = str(e)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    selected_ticker = None
    
    if ticker_input and ticker_input.strip():
        ticker_candidate = ticker_input.strip().upper()
        try:
            ticker = yf.Ticker(ticker_candidate)
            info = ticker.info
            if info:
                selected_ticker = ticker_candidate
        except:
            pass
    
    if not selected_ticker:
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'ticker[:\s]+([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|price|ticker))\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message.upper())
            if matches:
                for match in matches:
                    ticker_candidate = match[0] if isinstance(match, tuple) else match
                    try:
                        ticker = yf.Ticker(ticker_candidate)
                        info = ticker.info
                        if info:
                            selected_ticker = ticker_candidate
                            break
                    except:
                        continue
            if selected_ticker:
                break
    
    if not selected_ticker:
        try:
            extractor = TickerExtractor(apikey)
            selected_ticker = extractor.get_ticker_from_ai(message)
        except:
            pass
    
    if not selected_ticker:
        error_msg = ("I couldn't identify a stock ticker. Please:\n"
                    "1. Enter a ticker in the 'Stock Ticker' field (e.g., AAPL), or\n"
                    "2. Mention it in your message like '$AAPL' or 'AAPL stock'")
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    try:
        agent = FinancialAgent(api_key=apikey)
        df = agent.get_stock_data(selected_ticker, period_input)
        
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df = agent.create_technical_analysis(df)
            agent.df = df
            agent.selected_ticker = selected_ticker
            
            stock_plot = FinancialPlotter.create_stock_plot(df, selected_ticker)
            technical_plot = FinancialPlotter.create_technical_indicators_plot(df)
            volume_plot = FinancialPlotter.create_volume_analysis_plot(df)
            
            response = agent.get_response(
                f"{message}\nAnalyze {selected_ticker} based on this query:", 
                include_analysis=True
            )
            
            final_response = f"**{selected_ticker} Analysis**\n\n{response}"
            
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_response}
            ], stock_plot, technical_plot, volume_plot
        else:
            error_msg = f"Unable to fetch data for {selected_ticker}."
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ], None, None, None
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None

with gr.Blocks(title="Financial Analysis Assistant") as demo:
    gr.Markdown("# Financial Analysis Assistant\n*Powered by Mistral AI*")
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600, label="Chat")
            
            with gr.Row():
                ticker_input = gr.Textbox(
                    label="Stock Ticker",
                    placeholder="e.g., AAPL, TSLA, MSFT",
                    scale=1
                )
                period_input = gr.Dropdown(
                    choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                    label="Time Period",
                    value="1y",
                    scale=1
                )
            
            message = gr.Textbox(
                label="Your Message",
                placeholder="Type: 'Analyze Tesla' or 'What about $AAPL?'",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary", scale=2)
                clear = gr.Button("Clear", variant="secondary", scale=1)
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Stock Price"):
                    stock_plot = gr.Plot(label="Stock Price & Moving Averages")
                with gr.Tab("Technical Indicators"):
                    technical_plot = gr.Plot(label="RSI & MACD")
                with gr.Tab("Volume"):
                    volume_plot = gr.Plot(label="Volume Analysis")

    submit.click(
        fn=process_request,
        inputs=[message, ticker_input, period_input, chatbot],
        outputs=[chatbot, stock_plot, technical_plot, volume_plot]
    )
    
    clear.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[chatbot, stock_plot, technical_plot, volume_plot]
    )
    
    message.submit(
        fn=process_request,
        inputs=[message, ticker_input, period_input, chatbot],
        outputs=[chatbot, stock_plot, technical_plot, volume_plot]
    )

if __name__ == "__main__":
    print("=" * 50)
    print("Financial Analysis Assistant")
    print("=" * 50)
    
    try:
        apikey = load_api_key()
        print(f"✅ API Key loaded: {apikey[:10]}...")
        print("✅ Starting application...")
        demo.launch()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📝 Create a .env file with:")
        print("   MISTRAL_API_KEY=your_key_here")
        exit(1)
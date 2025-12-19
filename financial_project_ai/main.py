# main.py
import gradio as gr
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import re
import os
import time
from dotenv import load_dotenv
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from security_filters import SecurityFilter, SecureFinancialAgent
from data_security import SecureDataWrapper
from data_compliance import DataMinimizer, privacy_audit
import logging
from auth_interface import (
    create_auth_interface, 
    require_authentication, 
    check_session,
    current_session
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Initialiser le wrapper sécurisé
secure_data = SecureDataWrapper(cache_hours=24)


def load_api_key():
    """Load and validate API key from environment"""
    apikey = os.getenv('MISTRAL_API_KEY')
    
    if not apikey:
        raise ValueError(
            "❌ MISTRAL_API_KEY not found!\n"
            "Create a .env file with:\n"
            "MISTRAL_API_KEY=your_actual_key_here"
        )
    
    # Nettoyer la clé
    apikey = apikey.strip().strip('"').strip("'")
    
    # Valider avec SecurityFilter
    is_valid, error = SecurityFilter.validate_api_key(apikey)
    if not is_valid:
        raise ValueError(f"Invalid API key: {error}")
    
    return apikey


class SecureTickerExtractor:
    """Version sécurisée de TickerExtractor"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def extract_from_text(self, text):
        """Extract ticker from text using regex patterns"""
        # Valider l'entrée d'abord
        is_safe, cleaned_text = SecurityFilter.sanitize_input(text, "ticker_extraction")
        if not is_safe:
            logger.warning(f"Unsafe input in ticker extraction: {cleaned_text}")
            return None
        
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|equity))',
            r'\b([A-Z]{1,5})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                for potential_ticker in matches:
                    # Valider le ticker
                    is_valid, result = SecurityFilter.validate_ticker(potential_ticker)
                    if is_valid:
                        try:
                            ticker = yf.Ticker(result)
                            info = ticker.info
                            if info:
                                return result
                        except:
                            continue
        return None

    def get_ticker_from_ai(self, text):
        """Use Mistral AI to identify potential stock ticker from text"""
        # Valider l'entrée
        is_safe, cleaned_text = SecurityFilter.sanitize_input(text, "ai_extraction")
        if not is_safe:
            logger.warning("Unsafe input for AI extraction")
            return None
        
        prompt = f"Extract ONLY the stock ticker symbol from: {cleaned_text}. Return only the ticker, nothing else."
        
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
                
                # Valider le ticker extrait
                is_valid, validated_ticker = SecurityFilter.validate_ticker(potential_ticker)
                if not is_valid:
                    return None
                
                try:
                    ticker = yf.Ticker(validated_ticker)
                    info = ticker.info
                    if info:
                        return validated_ticker
                except:
                    return None
            return None
        except Exception as e:
            logger.error(f"AI extraction error: {str(e)}")
            return None


class SecureFinancialAgentExtended(SecureFinancialAgent):
    """Extension de SecureFinancialAgent avec analyse technique, cache sécurisé et conformité"""
    
    def __init__(self, api_key):
        super().__init__(api_key)
        self.ticker_extractor = SecureTickerExtractor(api_key)
        self.secure_data = secure_data  # Utiliser l'instance globale
    
    def get_stock_data(self, ticker, period="1y"):
        """Fetch stock data with data minimization and PII filtering"""
        # Valider le ticker
        is_valid, validated_ticker = SecurityFilter.validate_ticker(ticker)
        if not is_valid:
            logger.error(f"Invalid ticker: {validated_ticker}")
            return None
        
        # Valider la période
        is_valid, validated_period = SecurityFilter.validate_period(period)
        if not is_valid:
            logger.error(f"Invalid period: {validated_period}")
            return None
        
        # Fonction pour récupérer les données de yfinance
        def fetch_from_yfinance():
            try:
                logger.info(f"Fetching data for {validated_ticker} with period {validated_period}")
                
                stock = yf.Ticker(validated_ticker)
                df = stock.history(period=validated_period)
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {validated_ticker}")
                    # Essayer avec une période plus courte
                    logger.info("Trying with shorter period: 1mo")
                    df = stock.history(period="1mo")
                    
                    if df is None or df.empty:
                        logger.error(f"Still no data for {validated_ticker}")
                        return None
                
                # ✅ NOUVEAU : FILTRER LES COLONNES (minimisation)
                df, removed_columns = DataMinimizer.filter_dataframe_columns(df, validated_ticker)
                
                if df is None:
                    logger.error("DataFrame filtering failed")
                    return None
                
                if removed_columns:
                    logger.info(f"Removed {len(removed_columns)} unauthorized columns")
                
                # Vérifications de qualité
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing columns: {missing_columns}")
                    return None
                
                df = df.dropna(subset=required_columns)
                
                if df.empty:
                    logger.error("All rows contain NaN values")
                    return None
                
                logger.info(f"Successfully fetched {len(df)} rows for {validated_ticker}")
                logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                
                # ✅ AUDIT
                privacy_audit.log_data_access(
                    'market_data',
                    'read',
                    {'ticker': validated_ticker, 'rows': len(df), 'columns': list(df.columns)}
                )
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching from yfinance: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                return None
        
        # Utiliser le wrapper sécurisé avec cache et vérification d'intégrité
        return self.secure_data.get_stock_data_secure(
            validated_ticker, 
            validated_period, 
            fetch_from_yfinance
        )
    
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
    
    def get_response_with_analysis(self, message, ticker, df):
        """Get response with market insights and PII filtering"""
        self.df = df
        self.selected_ticker = ticker
        
        insights = self.generate_market_insights(df, ticker)
        full_message = f"{message}\n\n{insights}"
        
        # ✅ NOUVEAU : NETTOYER LE MESSAGE DE TOUTE INFO SENSIBLE
        cleaned_message, detected_pii = DataMinimizer.sanitize_text(full_message)
        
        if detected_pii:
            logger.warning(f"Detected and sanitized PII in message: {detected_pii}")
            privacy_audit.log_pii_detection('user_message', detected_pii)
        
        response = self.get_response(cleaned_message, include_analysis=False)
        
        # ✅ NETTOYER LA RÉPONSE AUSSI
        cleaned_response, detected_pii_response = DataMinimizer.sanitize_text(response)
        
        if detected_pii_response:
            logger.warning(f"Detected and sanitized PII in AI response: {detected_pii_response}")
            privacy_audit.log_pii_detection('ai_response', detected_pii_response)
            
        return cleaned_response


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
    """Clear all outputs"""
    return [], None, None, None


def secure_process_request_with_auth(message, ticker_input, period_input, history):
    """Process request with authentication check"""
    
    # 1. Vérifier l'authentification
    is_valid, username = check_session()
    
    if not is_valid:
        error_msg = "🔒 Authentication required. Please log in to use this service."
        return history + [
            {"role": "user", "content": message[:100] if message else ""},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 2. Logger l'activité de l'utilisateur
    logger.info(f"Request from authenticated user: {username}")
    
    # 3. Appeler la fonction d'origine
    return secure_process_request(message, ticker_input, period_input, history)


# Créer une interface combinée
# main.py - Partie à remplacer (vers la fin du fichier)

# ... (tout le code précédent reste identique jusqu'à la fonction secure_process_request) ...

def secure_process_request(message, ticker_input, period_input, history):
    """Process request with security filters"""
    
    if history is None:
        history = []
    
    # 1. Valider le message
    if not message or not message.strip():
        return history, None, None, None
    
    is_safe, cleaned_message = SecurityFilter.sanitize_input(message, "user_message")
    if not is_safe:
        error_msg = f"🛡️ Security Error: {cleaned_message}"
        logger.warning(f"Unsafe input detected: {message[:100]}")
        return history + [
            {"role": "user", "content": message[:100]},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 2. Détecter les injections de prompt
    is_injection, reason = SecurityFilter.detect_prompt_injection(cleaned_message)
    if is_injection:
        error_msg = f"🛡️ Security Alert: Prompt injection detected ({reason})"
        logger.warning(f"Prompt injection: {reason} - {message[:100]}")
        return history + [
            {"role": "user", "content": cleaned_message[:100]},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 3. Charger la clé API
    try:
        apikey = load_api_key()
    except Exception as e:
        error_msg = f"❌ Configuration Error: {str(e)}"
        return history + [
            {"role": "user", "content": cleaned_message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 4. Valider le ticker si fourni
    selected_ticker = None
    if ticker_input and ticker_input.strip():
        is_valid, result = SecurityFilter.validate_ticker(ticker_input)
        if not is_valid:
            error_msg = f"❌ Invalid ticker: {result}"
            return history + [
                {"role": "user", "content": cleaned_message},
                {"role": "assistant", "content": error_msg}
            ], None, None, None
        selected_ticker = result
    
    # 5. Valider la période
    is_valid, validated_period = SecurityFilter.validate_period(period_input)
    if not is_valid:
        error_msg = f"❌ Invalid period: {validated_period}"
        return history + [
            {"role": "user", "content": cleaned_message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 6. Extraction du ticker si non fourni
    if not selected_ticker:
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|price|ticker))\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_message.upper())
            if matches:
                for match in matches:
                    ticker_candidate = match[0] if isinstance(match, tuple) else match
                    is_valid, result = SecurityFilter.validate_ticker(ticker_candidate)
                    if is_valid:
                        selected_ticker = result
                        break
            if selected_ticker:
                break
    
    # 7. Extraction AI si toujours pas de ticker
    if not selected_ticker:
        try:
            extractor = SecureTickerExtractor(apikey)
            selected_ticker = extractor.get_ticker_from_ai(cleaned_message)
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
    
    # 8. Vérifier qu'on a un ticker
    if not selected_ticker:
        error_msg = ("I couldn't identify a stock ticker. Please:\n"
                    "1. Enter a ticker in the 'Stock Ticker' field (e.g., AAPL), or\n"
                    "2. Mention it in your message like '$AAPL' or 'AAPL stock'")
        return history + [
            {"role": "user", "content": cleaned_message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 9. Traiter la requête
    try:
        agent = SecureFinancialAgentExtended(api_key=apikey)
        df = agent.get_stock_data(selected_ticker, validated_period)
        
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df = agent.create_technical_analysis(df)
            
            # Créer les graphiques
            stock_plot = FinancialPlotter.create_stock_plot(df, selected_ticker)
            technical_plot = FinancialPlotter.create_technical_indicators_plot(df)
            volume_plot = FinancialPlotter.create_volume_analysis_plot(df)
            
            # Obtenir la réponse sécurisée
            response = agent.get_response_with_analysis(cleaned_message, selected_ticker, df)
            
            final_response = f"**{selected_ticker} Analysis**\n\n{response}"
            
            logger.info(f"Successful analysis for {selected_ticker}")
            
            return history + [
                {"role": "user", "content": cleaned_message},
                {"role": "assistant", "content": final_response}
            ], stock_plot, technical_plot, volume_plot
        else:
            error_msg = f"Unable to fetch data for {selected_ticker}"
            return history + [
                {"role": "user", "content": cleaned_message},
                {"role": "assistant", "content": error_msg}
            ], None, None, None
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        error_msg = "❌ An error occurred while processing your request"
        return history + [
            {"role": "user", "content": cleaned_message},
            {"role": "assistant", "content": error_msg}
        ], None, None, None


def secure_process_request_with_auth(message, ticker_input, period_input, history):
    """Process request with authentication check"""
    
    # 1. Vérifier l'authentification
    is_valid, username = check_session()
    
    if not is_valid:
        error_msg = "🔒 Authentication required. Please log in to use this service."
        return history + [
            {"role": "user", "content": message[:100] if message else ""},
            {"role": "assistant", "content": error_msg}
        ], None, None, None
    
    # 2. Logger l'activité de l'utilisateur
    logger.info(f"Request from authenticated user: {username}")
    
    # 3. Appeler la fonction d'origine
    return secure_process_request(message, ticker_input, period_input, history)


def test_mistral_api():
    """Test de connexion à l'API Mistral"""
    try:
        apikey = load_api_key()
        
        print("\n" + "="*60)
        print("🔍 Testing Mistral API Connection")
        print("="*60)
        
        headers = {
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json"
        }
        
        # Test 1: Liste des modèles disponibles
        print("\n📋 Test 1: Checking available models...")
        try:
            response = requests.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json()
                print(f"✅ API accessible - {len(models.get('data', []))} models available")
            else:
                print(f"⚠️  API returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
        
        # Test 2: Test de chat simple
        print("\n💬 Test 2: Testing chat endpoint...")
        try:
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Chat endpoint working")
            elif response.status_code == 502:
                print("⚠️  502 Bad Gateway - Mistral servers are experiencing issues")
                print("   This is temporary and usually resolves quickly")
            elif response.status_code == 503:
                print("⚠️  503 Service Unavailable - High load on Mistral servers")
            else:
                print(f"⚠️  Unexpected status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
        
        # Test 3: Vérifier le statut Mistral
        print("\n🌐 Test 3: Checking Mistral service status...")
        print("   Visit: https://status.mistral.ai/")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


def cleanup_cache():
    """Nettoyer le cache expiré au démarrage"""
    try:
        deleted = secure_data.clean_expired_cache()
        if deleted > 0:
            print(f"🧹 Cleaned {deleted} expired cache files")
        
        stats = secure_data.get_stats()
        print(f"📊 Cache stats: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")


# Créer une interface combinée
def create_complete_interface():
    """Créer l'interface complète avec authentification"""
    
    with gr.Blocks(title="🔐 Secure Financial Analysis") as complete_app:
        # Header
        gr.Markdown("""
        # 🔐 Secure Financial Analysis Assistant
        *Enterprise-Grade Security • Multi-Factor Authentication • End-to-End Encryption*
        """)
        
        # Afficher le statut de connexion
        with gr.Row():
            session_status = gr.Markdown("👤 **Status:** Not logged in")
            refresh_status = gr.Button("🔄 Refresh", size="sm")
        
        with gr.Tabs() as main_tabs:
            # Onglet Authentification
            with gr.Tab("🔐 Authentication"):
                auth_ui = create_auth_interface()
            
            # Onglet Application principale (protégé)
            with gr.Tab("📊 Financial Analysis"):
                gr.Markdown("""
                ### Market Analysis Tools
                *Requires authentication*
                """)
                
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
                            submit = gr.Button("🔍 Send", variant="primary", scale=2)
                            clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.Tab("Stock Price"):
                                stock_plot = gr.Plot(label="Stock Price & Moving Averages")
                            with gr.Tab("Technical Indicators"):
                                technical_plot = gr.Plot(label="RSI & MACD")
                            with gr.Tab("Volume"):
                                volume_plot = gr.Plot(label="Volume Analysis")
        
        # Event handlers
        def update_session_status():
            is_valid, username = check_session()
            if is_valid:
                return f"👤 **Status:** ✅ Logged in as **{username}**"
            else:
                return "👤 **Status:** ❌ Not logged in"
        
        refresh_status.click(
            fn=update_session_status,
            outputs=[session_status]
        )
        
        submit.click(
            fn=secure_process_request_with_auth,
            inputs=[message, ticker_input, period_input, chatbot],
            outputs=[chatbot, stock_plot, technical_plot, volume_plot]
        )
        
        clear.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[chatbot, stock_plot, technical_plot, volume_plot]
        )
        
        message.submit(
            fn=secure_process_request_with_auth,
            inputs=[message, ticker_input, period_input, chatbot],
            outputs=[chatbot, stock_plot, technical_plot, volume_plot]
        )
    
    return complete_app


# Point d'entrée principal
if __name__ == "__main__":
    print("=" * 60)
    print("🔐 Secure Financial Analysis Assistant")
    print("=" * 60)
    
    try:
        # Nettoyer le cache au démarrage
        cleanup_cache()
        
        # Nettoyer les sessions expirées
        from auth_interface import auth_manager
        auth_manager.session_manager.cleanup_expired_sessions()
        
        # Test de l'API
        test_mistral_api()
        
        # Charger et valider la clé API
        apikey = load_api_key()
        print(f"\n✅ API Key validated: {apikey[:10]}...")
        print("✅ Security filters enabled")
        print("✅ Encrypted cache enabled (24h retention)")
        print("✅ Data integrity verification enabled")
        print("✅ PII detection and filtering enabled")
        print("✅ GDPR compliance measures active")
        print("✅ Multi-Factor Authentication enabled")
        print("✅ JWT session management enabled")
        print("✅ Password policy enforcement enabled")
        
        print("\n🚀 Starting application...")
        
        # Lancer l'interface complète
        app = create_complete_interface()
        app.launch()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📝 Create a .env file with:")
        print("   MISTRAL_API_KEY=your_key_here")
        print("   JWT_SECRET=your_secret_here")
        exit(1)
    print("=" * 60)
    print("🔐 Secure Financial Analysis Assistant")
    print("=" * 60)
    
    try:
        # Nettoyer le cache au démarrage
        cleanup_cache()
        
        # Nettoyer les sessions expirées
        from auth_interface import auth_manager
        auth_manager.session_manager.cleanup_expired_sessions()
        
        # Test de l'API
        test_mistral_api()
        
        # Charger et valider la clé API
        apikey = load_api_key()
        print(f"\n✅ API Key validated: {apikey[:10]}...")
        print("✅ Security filters enabled")
        print("✅ Encrypted cache enabled (24h retention)")
        print("✅ Data integrity verification enabled")
        print("✅ PII detection and filtering enabled")
        print("✅ GDPR compliance measures active")
        print("✅ Multi-Factor Authentication enabled")
        print("✅ JWT session management enabled")
        print("✅ Password policy enforcement enabled")
        
        print("\n🚀 Starting application...")
        
        # Lancer l'interface complète
        app = create_complete_interface()
        app.launch()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📝 Create a .env file with:")
        print("   MISTRAL_API_KEY=your_key_here")
        print("   JWT_SECRET=your_secret_here")
        exit(1)
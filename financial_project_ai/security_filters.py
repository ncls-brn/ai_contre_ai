# security_filters.py
import re
import html
import requests  # ← AJOUTÉ
from typing import Tuple, List, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# security_filters.py - Méthode get_response améliorée

import time  # ← Ajouter cet import en haut du fichier

class SecureFinancialAgent:
    """Version sécurisée du FinancialAgent"""
    
    def __init__(self, api_key: str):
        # Valider la clé API
        is_valid, error = SecurityFilter.validate_api_key(api_key)
        if not is_valid:
            raise ValueError(f"Invalid API key: {error}")
        
        self.api_key = api_key.strip().strip('"').strip("'")
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.context = "You are a financial expert. Analyze technical indicators and provide clear, actionable insights."
        self.conversation_history = []
        self.max_retries = 3  # ← NOUVEAU
        self.retry_delay = 2  # ← NOUVEAU (secondes)
    
    def get_response(self, message: str, include_analysis: bool = True) -> str:
        """
        Obtenir une réponse sécurisée du modèle avec retry automatique
        
        Args:
            message: Message de l'utilisateur
            include_analysis: Inclure l'analyse technique
        
        Returns:
            str: Réponse nettoyée du modèle
        """
        # 1. Valider l'entrée
        is_safe, cleaned_message = SecurityFilter.sanitize_input(message, "user_message")
        if not is_safe:
            logger.error(f"Unsafe input detected: {cleaned_message}")
            return f"❌ Security Error: {cleaned_message}"
        
        # 2. Détecter les injections de prompt
        is_injection, reason = SecurityFilter.detect_prompt_injection(cleaned_message)
        if is_injection:
            logger.error(f"Prompt injection detected: {reason}")
            return "❌ Security Error: Your request contains potentially malicious patterns"
        
        # 3. Préparer les messages pour l'API
        messages = [{"role": "system", "content": self.context}]
        
        # Ajouter l'historique (limité aux 10 derniers messages)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": cleaned_message})
        
        # 4. Appeler l'API avec retry automatique
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "mistral-large-latest",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "top_p": 0.95,
                }
                
                logger.info(f"API request attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                # Succès
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result['choices'][0]['message']['content']
                    
                    # Nettoyer la sortie
                    cleaned_response = SecurityFilter.sanitize_output(assistant_message)
                    
                    # Mettre à jour l'historique
                    self.conversation_history.append({"role": "user", "content": cleaned_message})
                    self.conversation_history.append({"role": "assistant", "content": cleaned_response})
                    
                    # Limiter la taille de l'historique
                    if len(self.conversation_history) > 20:
                        self.conversation_history = self.conversation_history[-20:]
                    
                    logger.info("API request successful")
                    return cleaned_response
                
                # Erreurs qui ne nécessitent pas de retry
                elif response.status_code == 401:
                    logger.error("Authentication error (401)")
                    return (
                        "❌ Authentication Error (401)\n\n"
                        "Your Mistral API key is invalid or expired.\n\n"
                        "Steps to fix:\n"
                        "1. Go to https://console.mistral.ai/\n"
                        "2. Create a new API key\n"
                        "3. Update your .env file\n"
                        "4. Restart the application"
                    )
                
                elif response.status_code == 402:
                    logger.error("Payment required (402)")
                    return (
                        "❌ Payment Required (402)\n\n"
                        "Your Mistral account has insufficient credits.\n\n"
                        "Go to https://console.mistral.ai/billing to add credits."
                    )
                
                elif response.status_code == 400:
                    logger.error(f"Bad request (400): {response.text}")
                    return (
                        "❌ Bad Request (400)\n\n"
                        "The request format is invalid. Please try again."
                    )
                
                # Erreurs qui bénéficient d'un retry (502, 503, 504, 429)
                elif response.status_code in [502, 503, 504, 429]:
                    error_names = {
                        502: "Bad Gateway",
                        503: "Service Unavailable", 
                        504: "Gateway Timeout",
                        429: "Rate Limit Exceeded"
                    }
                    error_name = error_names.get(response.status_code, "Server Error")
                    
                    logger.warning(f"{error_name} ({response.status_code}) - Attempt {attempt + 1}/{self.max_retries}")
                    
                    # Si c'est la dernière tentative
                    if attempt == self.max_retries - 1:
                        return (
                            f"❌ {error_name} ({response.status_code})\n\n"
                            f"The Mistral API is temporarily unavailable.\n"
                            f"This usually resolves within a few minutes.\n\n"
                            f"What you can try:\n"
                            f"1. Wait 30 seconds and try again\n"
                            f"2. Check https://status.mistral.ai/ for service status\n"
                            f"3. If the problem persists, contact Mistral support"
                        )
                    
                    # Attendre avant le prochain essai (backoff exponentiel)
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Autres erreurs
                else:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return f"❌ API Error ({response.status_code}): Unable to process request"
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout - Attempt {attempt + 1}/{self.max_retries}")
                
                if attempt == self.max_retries - 1:
                    return "❌ Request timeout after multiple attempts. Please try again later."
                
                time.sleep(self.retry_delay)
                continue
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error - Attempt {attempt + 1}/{self.max_retries}")
                
                if attempt == self.max_retries - 1:
                    return "❌ Connection error. Please check your internet connection."
                
                time.sleep(self.retry_delay)
                continue
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                return "❌ Network error. Please check your internet connection."
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return "❌ An unexpected error occurred. Please try again."
        
        # Si on arrive ici, tous les retries ont échoué
        return "❌ Service temporarily unavailable. Please try again in a few moments."

class SecurityFilter:
    """Classe pour filtrer et valider les entrées/sorties"""
    
    # Patterns dangereux à détecter
    DANGEROUS_PATTERNS = [
        # Injection de commandes
        r';\s*(?:rm|del|format|shutdown|reboot)',
        r'\$\(.*?\)',
        r'`.*?`',
        r'\|\s*(?:bash|sh|cmd|powershell)',
        
        # Injection SQL (même si pas de DB, par précaution)
        r'(?:union|select|insert|update|delete|drop|create|alter)\s+(?:all|distinct|table|from|where)',
        
        # Path traversal
        r'\.\./|\.\.\\',
        r'(?:/etc/passwd|/etc/shadow|c:\\windows)',
        
        # Script injection
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on(?:load|error|click|mouseover)\s*=',
        
        # Prompt injection pour LLM
        r'ignore\s+(?:previous|all|above)\s+(?:instructions?|prompts?|context)',
        r'you\s+are\s+now\s+(?:a|an)',
        r'disregard\s+(?:previous|all|above)',
        r'system:\s*(?:override|bypass|ignore)',
        
        # Tentatives d'extraction de données sensibles
        r'(?:api[_\s-]?key|password|token|secret|credential)',
    ]
    
    # Caractères autorisés pour les tickers
    TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')
    
    # Longueur max des messages
    MAX_MESSAGE_LENGTH = 2000
    MAX_TICKER_LENGTH = 5
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """
        Valider un ticker boursier
        
        Returns:
            Tuple[bool, str]: (is_valid, cleaned_ticker or error_message)
        """
        if not ticker:
            return False, "Ticker is empty"
        
        # Nettoyer et convertir en majuscules
        ticker = ticker.strip().upper()
        
        # Vérifier la longueur
        if len(ticker) > SecurityFilter.MAX_TICKER_LENGTH:
            logger.warning(f"Ticker too long: {len(ticker)} chars")
            return False, f"Ticker too long (max {SecurityFilter.MAX_TICKER_LENGTH} chars)"
        
        # Vérifier le format
        if not SecurityFilter.TICKER_PATTERN.match(ticker):
            logger.warning(f"Invalid ticker format: {ticker}")
            return False, "Ticker must contain only uppercase letters (A-Z)"
        
        return True, ticker
    
    @staticmethod
    def validate_period(period: str) -> Tuple[bool, str]:
        """
        Valider une période de temps
        
        Returns:
            Tuple[bool, str]: (is_valid, period or error_message)
        """
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        
        if period not in valid_periods:
            logger.warning(f"Invalid period: {period}")
            return False, f"Period must be one of: {', '.join(valid_periods)}"
        
        return True, period
    
    @staticmethod
    def sanitize_input(text: str, context: str = "message") -> Tuple[bool, str]:
        """
        Nettoyer et valider une entrée utilisateur
        
        Args:
            text: Le texte à valider
            context: Le contexte (message, ticker, etc.)
        
        Returns:
            Tuple[bool, str]: (is_safe, cleaned_text or error_message)
        """
        if not text:
            return False, "Input is empty"
        
        # Vérifier la longueur
        if len(text) > SecurityFilter.MAX_MESSAGE_LENGTH:
            logger.warning(f"Input too long: {len(text)} chars in {context}")
            return False, f"Input too long (max {SecurityFilter.MAX_MESSAGE_LENGTH} chars)"
        
        # Nettoyer les espaces
        text = text.strip()
        
        # Détecter les patterns dangereux
        for pattern in SecurityFilter.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in {context}: {pattern}")
                return False, "Input contains potentially malicious content"
        
        # Vérifier les caractères de contrôle
        if any(ord(char) < 32 and char not in ['\n', '\r', '\t'] for char in text):
            logger.warning(f"Control characters detected in {context}")
            return False, "Input contains invalid control characters"
        
        # Échapper les caractères HTML
        text = html.escape(text)
        
        return True, text
    
    @staticmethod
    def sanitize_output(text: str) -> str:
        """
        Nettoyer la sortie du modèle LLM
        
        Args:
            text: La réponse du modèle
        
        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""
        
        # Supprimer les balises script
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Supprimer les event handlers JavaScript
        text = re.sub(r'on\w+\s*=\s*["\'].*?["\']', '', text, flags=re.IGNORECASE)
        
        # Supprimer les URLs javascript:
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Supprimer les tentatives de path traversal
        text = re.sub(r'\.\./|\.\.\\', '', text)
        
        # Limiter les caractères répétés (potentiel DoS)
        text = re.sub(r'(.)\1{50,}', r'\1' * 50, text)
        
        # Échapper HTML
        text = html.escape(text)
        
        # Limiter la longueur totale
        if len(text) > 10000:
            text = text[:10000] + "\n\n[Response truncated for safety]"
        
        return text
    
    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, str]:
        """
        Valider le format d'une clé API
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not api_key:
            return False, "API key is empty"
        
        # Nettoyer
        api_key = api_key.strip().strip('"').strip("'")
        
        # Vérifier la longueur (les clés API ont généralement >20 chars)
        if len(api_key) < 20:
            return False, "API key seems too short"
        
        # Vérifier qu'elle ne contient pas d'espaces
        if ' ' in api_key:
            return False, "API key contains spaces"
        
        # Vérifier les caractères autorisés (alphanumériques + - et _)
        if not re.match(r'^[A-Za-z0-9\-_]+$', api_key):
            return False, "API key contains invalid characters"
        
        return True, ""
    
    @staticmethod
    def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
        """
        Détecter les tentatives d'injection de prompt
        
        Returns:
            Tuple[bool, Optional[str]]: (is_injection, reason)
        """
        injection_patterns = [
            (r'ignore\s+(?:previous|all|above)\s+(?:instructions?|prompts?)', "Ignore instruction attempt"),
            (r'you\s+are\s+now\s+(?:a|an)\s+\w+', "Role change attempt"),
            (r'disregard\s+(?:previous|all|above)', "Disregard instruction attempt"),
            (r'system:\s*(?:override|bypass|ignore)', "System override attempt"),
            (r'(?:act|behave|pretend)\s+(?:as|like)\s+(?:if|you)', "Behavior modification attempt"),
            (r'\[SYSTEM\]|\[ADMIN\]|\[ROOT\]', "System role impersonation"),
            (r'reveal\s+(?:your|the)\s+(?:prompt|instructions?|system)', "Prompt extraction attempt"),
        ]
        
        for pattern, reason in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Prompt injection detected: {reason}")
                return True, reason
        
        return False, None
    
    @staticmethod
    def rate_limit_check(user_id: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
        """
        Vérification simple du rate limiting (à implémenter avec Redis en production)
        
        Returns:
            bool: True si la requête est autorisée
        """
        # TODO: Implémenter avec Redis ou une base de données
        # Pour l'instant, on retourne True
        return True


class SecureFinancialAgent:
    """Version sécurisée du FinancialAgent"""
    
    def __init__(self, api_key: str):
        # Valider la clé API
        is_valid, error = SecurityFilter.validate_api_key(api_key)
        if not is_valid:
            raise ValueError(f"Invalid API key: {error}")
        
        self.api_key = api_key.strip().strip('"').strip("'")
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.context = "You are a financial expert. Analyze technical indicators and provide clear, actionable insights."
        self.conversation_history = []
    
    def get_response(self, message: str, include_analysis: bool = True) -> str:
        """
        Obtenir une réponse sécurisée du modèle
        
        Args:
            message: Message de l'utilisateur
            include_analysis: Inclure l'analyse technique
        
        Returns:
            str: Réponse nettoyée du modèle
        """
        # 1. Valider l'entrée
        is_safe, cleaned_message = SecurityFilter.sanitize_input(message, "user_message")
        if not is_safe:
            logger.error(f"Unsafe input detected: {cleaned_message}")
            return f"❌ Security Error: {cleaned_message}"
        
        # 2. Détecter les injections de prompt
        is_injection, reason = SecurityFilter.detect_prompt_injection(cleaned_message)
        if is_injection:
            logger.error(f"Prompt injection detected: {reason}")
            return "❌ Security Error: Your request contains potentially malicious patterns"
        
        # 3. Préparer les messages pour l'API
        messages = [{"role": "system", "content": self.context}]
        
        # Ajouter l'historique (limité aux 10 derniers messages pour éviter les abus)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": cleaned_message})
        
        # 4. Appeler l'API avec timeout et gestion d'erreurs
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.95,
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                
                # 5. Nettoyer la sortie du modèle
                cleaned_response = SecurityFilter.sanitize_output(assistant_message)
                
                # 6. Mettre à jour l'historique (nettoyé)
                self.conversation_history.append({"role": "user", "content": cleaned_message})
                self.conversation_history.append({"role": "assistant", "content": cleaned_response})
                
                # 7. Limiter la taille de l'historique
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                return cleaned_response
            
            elif response.status_code == 401:
                return (
                    "❌ Authentication Error (401)\n\n"
                    "Your Mistral API key is invalid or expired.\n\n"
                    "Steps to fix:\n"
                    "1. Go to https://console.mistral.ai/\n"
                    "2. Navigate to 'API Keys'\n"
                    "3. Create a new key\n"
                    "4. Update your .env file:\n"
                    "   MISTRAL_API_KEY=your_new_key\n"
                    "5. Restart the application"
                )
            
            elif response.status_code == 402:
                return (
                    "❌ Payment Required (402)\n\n"
                    "Your Mistral account has insufficient credits.\n\n"
                    "Go to https://console.mistral.ai/billing to add credits."
                )
            
            elif response.status_code == 429:
                return (
                    "❌ Rate Limit Exceeded (429)\n\n"
                    "You've made too many requests.\n"
                    "Please wait a moment and try again."
                )
            
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return f"❌ API Error: Unable to process request (Status {response.status_code})"
                
        except requests.exceptions.Timeout:
            logger.error("API timeout")
            return "❌ Request timeout. Please try again."
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return "❌ Connection error. Please check your internet connection."
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "❌ An unexpected error occurred. Please try again."
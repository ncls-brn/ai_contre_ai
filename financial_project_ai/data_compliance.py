# data_compliance.py
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataMinimizer:
    """
    Classe pour minimiser et filtrer les données selon les principes RGPD
    """
    
    # Patterns d'informations sensibles à détecter et supprimer
    SENSITIVE_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
        'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        'address': r'\b\d{1,5}\s+[\w\s]{1,50}(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'api_key_pattern': r'(?i)(?:api[_-]?key|apikey|access[_-]?token|secret[_-]?key)[\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    }
    
    # Colonnes autorisées pour les données financières (liste blanche)
    ALLOWED_FINANCIAL_COLUMNS = {
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Adj Close', 'Dividends', 'Stock Splits',
        'Date', 'Datetime', 'Timestamp'
    }
    
    # Colonnes à exclure absolument (liste noire)
    BLACKLISTED_COLUMNS = {
        'user_id', 'customer_id', 'account_id', 'account_number',
        'user_name', 'customer_name', 'full_name', 'name',
        'email', 'phone', 'address', 'ssn', 'tax_id',
        'password', 'token', 'session_id', 'auth_token',
        'credit_card', 'card_number', 'cvv', 'expiry',
        'bank_account', 'routing_number', 'swift_code',
        'ip_address', 'mac_address', 'device_id',
        'birth_date', 'date_of_birth', 'age', 'gender',
        'nationality', 'passport', 'license_number'
    }
    
    @staticmethod
    def filter_dataframe_columns(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filtrer un DataFrame pour ne garder que les colonnes autorisées
        
        Args:
            df: DataFrame à filtrer
            ticker: Symbole du ticker (pour logging)
        
        Returns:
            Tuple[DataFrame, List[str]]: DataFrame filtré et liste des colonnes supprimées
        """
        if df is None or df.empty:
            return df, []
        
        original_columns = set(df.columns)
        removed_columns = []
        
        # 1. Supprimer les colonnes de la liste noire
        blacklisted_found = original_columns.intersection(DataMinimizer.BLACKLISTED_COLUMNS)
        if blacklisted_found:
            logger.warning(f"Found blacklisted columns in {ticker}: {blacklisted_found}")
            df = df.drop(columns=list(blacklisted_found))
            removed_columns.extend(blacklisted_found)
        
        # 2. Garder uniquement les colonnes autorisées
        current_columns = set(df.columns)
        unauthorized = current_columns - DataMinimizer.ALLOWED_FINANCIAL_COLUMNS
        
        if unauthorized:
            logger.info(f"Removing unauthorized columns from {ticker}: {unauthorized}")
            df = df.drop(columns=list(unauthorized))
            removed_columns.extend(unauthorized)
        
        # 3. Vérifier qu'il reste des colonnes essentielles
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        remaining = set(df.columns)
        
        if not required_columns.issubset(remaining):
            logger.error(f"Missing required columns after filtering for {ticker}")
            return None, removed_columns
        
        if removed_columns:
            logger.info(f"Removed {len(removed_columns)} columns from {ticker}: {removed_columns}")
        
        return df, removed_columns
    
    @staticmethod
    def sanitize_text(text: str) -> Tuple[str, List[str]]:
        """
        Nettoyer un texte de toutes les informations sensibles
        
        Args:
            text: Texte à nettoyer
        
        Returns:
            Tuple[str, List[str]]: Texte nettoyé et types d'infos détectées
        """
        if not text:
            return text, []
        
        detected_types = []
        cleaned_text = text
        
        for pattern_type, pattern in DataMinimizer.SENSITIVE_PATTERNS.items():
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            if matches:
                detected_types.append(pattern_type)
                # Remplacer par [REDACTED]
                cleaned_text = re.sub(pattern, f'[{pattern_type.upper()}_REDACTED]', cleaned_text, flags=re.IGNORECASE)
                logger.warning(f"Detected and redacted {pattern_type}: {len(matches)} occurrences")
        
        return cleaned_text, detected_types
    
    @staticmethod
    def minimize_cache_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimiser les données avant mise en cache
        
        Args:
            data: Données à minimiser
        
        Returns:
            Dict: Données minimisées
        """
        if not data:
            return data
        
        minimized = {}
        
        # Ne garder que les colonnes autorisées
        for key, value in data.items():
            if key in DataMinimizer.ALLOWED_FINANCIAL_COLUMNS or key == 'index':
                minimized[key] = value
            else:
                logger.debug(f"Excluded column from cache: {key}")
        
        return minimized
    
    @staticmethod
    def validate_no_pii(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valider qu'aucune donnée personnelle n'est présente
        
        Args:
            data: Données à valider
        
        Returns:
            Tuple[bool, List[str]]: (est_valide, liste_des_violations)
        """
        violations = []
        
        # Vérifier les clés du dictionnaire
        data_keys = set(data.keys()) if isinstance(data, dict) else set()
        blacklisted_keys = data_keys.intersection(DataMinimizer.BLACKLISTED_COLUMNS)
        
        if blacklisted_keys:
            violations.append(f"Blacklisted keys found: {blacklisted_keys}")
        
        # Vérifier les valeurs (si ce sont des strings)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    _, detected = DataMinimizer.sanitize_text(value)
                    if detected:
                        violations.append(f"Sensitive data in key '{key}': {detected}")
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            logger.error(f"PII validation failed: {violations}")
        
        return is_valid, violations


class DataRetentionPolicy:
    """
    Politique de rétention des données
    """
    
    # Durées de rétention par type de données (en jours)
    RETENTION_PERIODS = {
        'market_data': 1,      # Données de marché : 24h
        'analysis': 0,         # Analyses : pas de stockage
        'conversation': 0,     # Conversations : pas de stockage
        'cache': 1,           # Cache : 24h
        'logs': 7,            # Logs : 7 jours
    }
    
    @staticmethod
    def get_retention_period(data_type: str) -> int:
        """
        Obtenir la période de rétention pour un type de données
        
        Args:
            data_type: Type de données
        
        Returns:
            int: Nombre de jours de rétention
        """
        return DataRetentionPolicy.RETENTION_PERIODS.get(data_type, 0)
    
    @staticmethod
    def should_retain(data_type: str) -> bool:
        """
        Déterminer si un type de données doit être conservé
        
        Args:
            data_type: Type de données
        
        Returns:
            bool: True si doit être conservé
        """
        return DataRetentionPolicy.get_retention_period(data_type) > 0


class PrivacyAudit:
    """
    Audit de conformité à la vie privée
    """
    
    def __init__(self):
        self.audit_log = []
    
    def log_data_access(self, data_type: str, action: str, details: Dict[str, Any]):
        """
        Logger un accès aux données
        
        Args:
            data_type: Type de données accédées
            action: Action effectuée (read, write, delete)
            details: Détails supplémentaires
        """
        from datetime import datetime
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'action': action,
            'details': details
        }
        
        self.audit_log.append(entry)
        logger.info(f"Audit: {action} on {data_type}")
    
    def log_pii_detection(self, location: str, pii_types: List[str]):
        """
        Logger une détection de données personnelles
        
        Args:
            location: Où les données ont été détectées
            pii_types: Types de données personnelles détectées
        """
        from datetime import datetime
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'pii_detection',
            'location': location,
            'pii_types': pii_types,
            'action_taken': 'redacted'
        }
        
        self.audit_log.append(entry)
        logger.warning(f"PII detected and redacted at {location}: {pii_types}")
    
    def get_audit_report(self) -> Dict[str, Any]:
        """
        Obtenir un rapport d'audit
        
        Returns:
            Dict: Rapport d'audit
        """
        return {
            'total_entries': len(self.audit_log),
            'entries': self.audit_log[-100:]  # Dernières 100 entrées
        }


# Instance globale pour l'audit
privacy_audit = PrivacyAudit()
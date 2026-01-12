# data_security.py
import hashlib
import hmac
import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # ← CORRIGÉ
import logging
from data_compliance import DataMinimizer, DataRetentionPolicy, privacy_audit

logger = logging.getLogger(__name__)


# data_security.py - Ajouter ces imports en haut


# Modifier la méthode save_to_cache dans SecureDataManager
class SecureDataManager:
    # ... code existant ...
    
    def save_to_cache(self, ticker: str, period: str, data: Dict[str, Any]) -> bool:
        """
        Sauvegarder les données dans le cache chiffré avec minimisation
        """
        try:
            # 1. MINIMISER LES DONNÉES
            minimized_data = DataMinimizer.minimize_cache_data(data)
            
            # 2. VALIDER QU'IL N'Y A PAS DE PII
            is_valid, violations = DataMinimizer.validate_no_pii(minimized_data)
            if not is_valid:
                logger.error(f"Cannot cache data with PII violations: {violations}")
                privacy_audit.log_pii_detection('cache_attempt', violations)
                return False
            
            # 3. VÉRIFIER LA POLITIQUE DE RÉTENTION
            if not DataRetentionPolicy.should_retain('market_data'):
                logger.info("Data type should not be retained per policy")
                return False
            
            cache_key = self._get_cache_key(ticker, period)
            cache_file = self.cache_dir / f"{cache_key}.enc"
            
            # Préparer les métadonnées (minimisées)
            cache_data = {
                'ticker': ticker,
                'period': period,
                'timestamp': datetime.now().isoformat(),
                'data': minimized_data,
                'retention_days': DataRetentionPolicy.get_retention_period('market_data')
            }
            
            # Sérialiser en JSON
            json_data = json.dumps(cache_data).encode()
            
            # Calculer le HMAC
            data_hmac = self._compute_hmac(json_data)
            
            # Chiffrer les données
            encrypted_data = self.cipher_suite.encrypt(json_data)
            
            # Sauvegarder avec le HMAC
            with open(cache_file, 'wb') as f:
                f.write(data_hmac.encode() + b'\n' + encrypted_data)
            
            os.chmod(cache_file, 0o600)
            
            # 4. LOGGER L'AUDIT
            privacy_audit.log_data_access(
                'market_data',
                'write',
                {'ticker': ticker, 'period': period, 'size_bytes': len(encrypted_data)}
            )
            
            logger.info(f"Cached minimized data for {ticker} ({period})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
        
class SecureDataManager:
    """Gestionnaire sécurisé pour les données financières"""
    
    def __init__(self, cache_dir: str = ".cache", max_cache_age_hours: int = 24):
        """
        Initialize secure data manager
        
        Args:
            cache_dir: Répertoire pour le cache chiffré
            max_cache_age_hours: Durée maximale de conservation du cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, mode=0o700)  # Permissions restrictives
        
        self.max_cache_age = timedelta(hours=max_cache_age_hours)
        self.cipher_suite = self._initialize_encryption()
        
        # Clé pour HMAC (vérification d'intégrité)
        self.hmac_key = self._get_or_create_hmac_key()
        
        logger.info(f"Secure data manager initialized with cache dir: {self.cache_dir}")
    
    def _initialize_encryption(self) -> Fernet:
        """
        Initialiser le système de chiffrement
        
        Returns:
            Fernet: Objet de chiffrement
        """
        key_file = self.cache_dir / ".encryption_key"
        
        # Charger ou créer la clé de chiffrement
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
            logger.info("Loaded existing encryption key")
        else:
            # Générer une nouvelle clé
            key = Fernet.generate_key()
            
            # Sauvegarder avec permissions restrictives
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Lecture/écriture propriétaire uniquement
            
            logger.info("Generated new encryption key")
        
        return Fernet(key)
    
    def _get_or_create_hmac_key(self) -> bytes:
        """
        Obtenir ou créer la clé HMAC pour vérification d'intégrité
        
        Returns:
            bytes: Clé HMAC
        """
        hmac_key_file = self.cache_dir / ".hmac_key"
        
        if hmac_key_file.exists():
            with open(hmac_key_file, 'rb') as f:
                return f.read()
        else:
            # Générer une clé aléatoire de 32 bytes
            key = os.urandom(32)
            with open(hmac_key_file, 'wb') as f:
                f.write(key)
            os.chmod(hmac_key_file, 0o600)
            return key
    
    def _compute_hash(self, data: bytes) -> str:
        """
        Calculer le hash SHA-256 des données
        
        Args:
            data: Données à hasher
        
        Returns:
            str: Hash hexadécimal
        """
        return hashlib.sha256(data).hexdigest()
    
    def _compute_hmac(self, data: bytes) -> str:
        """
        Calculer le HMAC des données pour vérification d'intégrité
        
        Args:
            data: Données à signer
        
        Returns:
            str: HMAC hexadécimal
        """
        return hmac.new(self.hmac_key, data, hashlib.sha256).hexdigest()
    
    def _verify_hmac(self, data: bytes, expected_hmac: str) -> bool:
        """
        Vérifier l'intégrité des données avec HMAC
        
        Args:
            data: Données à vérifier
            expected_hmac: HMAC attendu
        
        Returns:
            bool: True si les données sont intègres
        """
        computed_hmac = self._compute_hmac(data)
        return hmac.compare_digest(computed_hmac, expected_hmac)
    
    def _get_cache_key(self, ticker: str, period: str) -> str:
        """
        Générer une clé de cache unique
        
        Args:
            ticker: Symbole boursier
            period: Période des données
        
        Returns:
            str: Clé de cache
        """
        raw_key = f"{ticker}_{period}".encode()
        return self._compute_hash(raw_key)
    
    def save_to_cache(self, ticker: str, period: str, data: Dict[str, Any]) -> bool:
        """
        Sauvegarder les données dans le cache chiffré
        
        Args:
            ticker: Symbole boursier
            period: Période des données
            data: Données à sauvegarder
        
        Returns:
            bool: True si succès
        """
        try:
            cache_key = self._get_cache_key(ticker, period)
            cache_file = self.cache_dir / f"{cache_key}.enc"
            
            # Préparer les métadonnées
            cache_data = {
                'ticker': ticker,
                'period': period,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # Sérialiser en JSON
            json_data = json.dumps(cache_data).encode()
            
            # Calculer le HMAC
            data_hmac = self._compute_hmac(json_data)
            
            # Chiffrer les données
            encrypted_data = self.cipher_suite.encrypt(json_data)
            
            # Sauvegarder avec le HMAC
            with open(cache_file, 'wb') as f:
                # Format: HMAC (64 bytes hex) + nouvelle ligne + données chiffrées
                f.write(data_hmac.encode() + b'\n' + encrypted_data)
            
            # Permissions restrictives
            os.chmod(cache_file, 0o600)
            
            logger.info(f"Cached data for {ticker} ({period})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def load_from_cache(self, ticker: str, period: str) -> Optional[Dict[str, Any]]:
        """
        Charger les données depuis le cache chiffré avec vérification d'intégrité
        
        Args:
            ticker: Symbole boursier
            period: Période des données
        
        Returns:
            Optional[Dict]: Données déchiffrées ou None
        """
        try:
            cache_key = self._get_cache_key(ticker, period)
            cache_file = self.cache_dir / f"{cache_key}.enc"
            
            if not cache_file.exists():
                logger.debug(f"No cache found for {ticker} ({period})")
                return None
            
            # Lire le fichier
            with open(cache_file, 'rb') as f:
                content = f.read()
            
            # Séparer le HMAC et les données chiffrées
            parts = content.split(b'\n', 1)
            if len(parts) != 2:
                logger.warning("Invalid cache file format")
                return None
            
            stored_hmac = parts[0].decode()
            encrypted_data = parts[1]
            
            # Déchiffrer
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Vérifier l'intégrité avec HMAC
            if not self._verify_hmac(decrypted_data, stored_hmac):
                logger.error(f"HMAC verification failed for {ticker} - data may be corrupted")
                # Supprimer le fichier corrompu
                cache_file.unlink()
                return None
            
            # Désérialiser
            cache_data = json.loads(decrypted_data.decode())
            
            # Vérifier l'âge du cache
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            age = datetime.now() - cached_time
            
            if age > self.max_cache_age:
                logger.info(f"Cache expired for {ticker} (age: {age})")
                cache_file.unlink()
                return None
            
            logger.info(f"Loaded cached data for {ticker} (age: {age})")
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def verify_yfinance_integrity(self, df_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifier l'intégrité des données Yahoo Finance
        
        Args:
            df_data: Données du DataFrame en format dict
        
        Returns:
            Tuple[bool, str]: (is_valid, hash)
        """
        try:
            # Normaliser les données pour un hachage cohérent
            normalized_data = json.dumps(df_data, sort_keys=True).encode()
            
            # Calculer le hash
            data_hash = self._compute_hash(normalized_data)
            
            # Vérifications de cohérence basiques
            if not df_data:
                logger.warning("Empty data received")
                return False, ""
            
            # Vérifier que les colonnes essentielles sont présentes
            required_keys = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Si les données sont structurées par colonne
            if isinstance(df_data, dict):
                for key in required_keys:
                    if key not in df_data:
                        logger.warning(f"Missing required column: {key}")
                        return False, ""
            
            logger.info(f"Data integrity verified - Hash: {data_hash[:16]}...")
            return True, data_hash
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            return False, ""
    
    def clean_old_cache(self) -> int:
        """
        Nettoyer les fichiers de cache expirés
        
        Returns:
            int: Nombre de fichiers supprimés
        """
        deleted_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.enc"):
                try:
                    # Vérifier l'âge du fichier
                    file_age = datetime.now() - datetime.fromtimestamp(
                        cache_file.stat().st_mtime
                    )
                    
                    if file_age > self.max_cache_age:
                        cache_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted expired cache file: {cache_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Error processing cache file {cache_file}: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned {deleted_count} expired cache files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            return 0
    
    def clear_all_cache(self) -> bool:
        """
        Supprimer tous les fichiers de cache
        
        Returns:
            bool: True si succès
        """
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.enc"):
                cache_file.unlink()
                count += 1
            
            logger.info(f"Cleared all cache ({count} files)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtenir des statistiques sur le cache
        
        Returns:
            Dict: Statistiques du cache
        """
        try:
            cache_files = list(self.cache_dir.glob("*.enc"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats = {
                'total_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
                'max_age_hours': self.max_cache_age.total_seconds() / 3600
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class SecureDataWrapper:
    """Wrapper pour intégrer la sécurité des données avec yfinance"""
    
    def __init__(self, cache_hours: int = 24):
        """
        Initialize secure data wrapper
        
        Args:
            cache_hours: Durée de conservation du cache en heures
        """
        self.data_manager = SecureDataManager(max_cache_age_hours=cache_hours)
        logger.info("Secure data wrapper initialized")
    
    def get_stock_data_secure(self, ticker: str, period: str, fetch_function):
        """
        Récupérer les données avec cache sécurisé et vérification d'intégrité
        
        Args:
            ticker: Symbole boursier
            period: Période des données
            fetch_function: Fonction pour récupérer les données (si pas en cache)
        
        Returns:
            DataFrame ou None
        """
        # 1. Essayer de charger depuis le cache
        cached_data = self.data_manager.load_from_cache(ticker, period)
        
        if cached_data:
            logger.info(f"Using cached data for {ticker}")
            
            # Vérifier l'intégrité des données en cache
            is_valid, data_hash = self.data_manager.verify_yfinance_integrity(cached_data)
            
            if not is_valid:
                logger.warning("Cached data failed integrity check, fetching fresh data")
            else:
                # Reconstruire le DataFrame depuis les données en cache
                import pandas as pd
                try:
                    df = pd.DataFrame(cached_data)
                    if 'index' in cached_data:
                        df.index = pd.to_datetime(cached_data['index'])
                    return df
                except Exception as e:
                    logger.error(f"Error reconstructing DataFrame from cache: {e}")
        
        # 2. Récupérer les données fraîches
        logger.info(f"Fetching fresh data for {ticker}")
        df = fetch_function()
        
        if df is None or df.empty:
            return None
        
        # 3. Convertir en dict pour le cache
        df_dict = df.to_dict()
        df_dict['index'] = df.index.astype(str).tolist()
        
        # 4. Vérifier l'intégrité des données fraîches
        is_valid, data_hash = self.data_manager.verify_yfinance_integrity(df_dict)
        
        if not is_valid:
            logger.warning("Fresh data failed integrity check")
            return df  # Retourner quand même les données mais sans les mettre en cache
        
        # 5. Sauvegarder dans le cache chiffré
        self.data_manager.save_to_cache(ticker, period, df_dict)
        
        return df
    
    def clean_expired_cache(self) -> int:
        """Nettoyer le cache expiré"""
        return self.data_manager.clean_old_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du cache"""
        return self.data_manager.get_cache_stats()
# test_data_security.py
import unittest
import tempfile
import shutil
from pathlib import Path
from data_security import SecureDataManager
import time

class TestSecureDataManager(unittest.TestCase):
    
    def setUp(self):
        """Créer un répertoire temporaire pour les tests"""
        self.test_dir = tempfile.mkdtemp()
        self.manager = SecureDataManager(cache_dir=self.test_dir, max_cache_age_hours=1)
    
    def tearDown(self):
        """Nettoyer le répertoire temporaire"""
        shutil.rmtree(self.test_dir)
    
    def test_save_and_load_cache(self):
        """Test de sauvegarde et chargement du cache"""
        test_data = {
            'Open': [100, 101, 102],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }
        
        # Sauvegarder
        success = self.manager.save_to_cache('AAPL', '1y', test_data)
        self.assertTrue(success)
        
        # Charger
        loaded_data = self.manager.load_from_cache('AAPL', '1y')
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data, test_data)
    
    def test_cache_expiration(self):
        """Test de l'expiration du cache"""
        # Créer un manager avec un cache très court
        short_cache = SecureDataManager(
            cache_dir=self.test_dir, 
            max_cache_age_hours=0.0001  # ~0.36 secondes
        )
        
        test_data = {'test': 'data'}
        short_cache.save_to_cache('TEST', '1d', test_data)
        
        # Attendre l'expiration
        time.sleep(1)
        
        # Le cache devrait être expiré
        loaded_data = short_cache.load_from_cache('TEST', '1d')
        self.assertIsNone(loaded_data)
    
    def test_integrity_verification(self):
        """Test de vérification d'intégrité"""
        test_data = {
            'Open': [100, 101],
            'Close': [101, 102]
        }
        
        is_valid, hash_value = self.manager.verify_yfinance_integrity(test_data)
        self.assertTrue(is_valid)
        self.assertTrue(len(hash_value) > 0)
    
    def test_cache_corruption_detection(self):
        """Test de détection de corruption"""
        test_data = {'test': 'data'}
        self.manager.save_to_cache('CORRUPT', '1d', test_data)
        
        # Corrompre le fichier
        cache_files = list(Path(self.test_dir).glob("*.enc"))
        if cache_files:
            with open(cache_files[0], 'ab') as f:
                f.write(b'CORRUPTED_DATA')
        
        # Le chargement devrait échouer
        loaded_data = self.manager.load_from_cache('CORRUPT', '1d')
        self.assertIsNone(loaded_data)
    
    def test_clean_old_cache(self):
        """Test de nettoyage du cache"""
        # Créer plusieurs entrées
        for i in range(5):
            self.manager.save_to_cache(f'TEST{i}', '1d', {'data': i})
        
        # Vérifier qu'elles existent
        cache_files = list(Path(self.test_dir).glob("*.enc"))
        self.assertEqual(len(cache_files), 5)
        
        # Nettoyer (ne devrait rien supprimer car récent)
        deleted = self.manager.clean_old_cache()
        self.assertEqual(deleted, 0)
    
    def test_encryption(self):
        """Test du chiffrement"""
        test_data = {'sensitive': 'data', 'values': [1, 2, 3]}
        self.manager.save_to_cache('SECRET', '1d', test_data)
        
        # Lire le fichier brut
        cache_files = list(Path(self.test_dir).glob("*.enc"))
        with open(cache_files[0], 'rb') as f:
            raw_content = f.read()
        
        # Vérifier que les données sensibles ne sont pas en clair
        self.assertNotIn(b'sensitive', raw_content)
        self.assertNotIn(b'data', raw_content)

if __name__ == '__main__':
    unittest.main()
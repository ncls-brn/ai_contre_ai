# test_compliance.py
import unittest
import pandas as pd
from data_compliance import DataMinimizer, DataRetentionPolicy, PrivacyAudit

class TestDataCompliance(unittest.TestCase):
    
    def test_email_detection(self):
        """Test de détection d'email"""
        text = "Contact me at john.doe@example.com"
        cleaned, detected = DataMinimizer.sanitize_text(text)
        
        self.assertIn('email', detected)
        self.assertNotIn('john.doe@example.com', cleaned)
        self.assertIn('[EMAIL_REDACTED]', cleaned)
    
    def test_phone_detection(self):
        """Test de détection de téléphone"""
        text = "Call me at 555-123-4567"
        cleaned, detected = DataMinimizer.sanitize_text(text)
        
        self.assertIn('phone', detected)
        self.assertNotIn('555-123-4567', cleaned)
    
    def test_dataframe_filtering(self):
        """Test de filtrage de DataFrame"""
        df = pd.DataFrame({
            'Open': [100, 101],
            'Close': [101, 102],
            'user_id': [123, 456],  # Doit être supprimé
            'email': ['test@test.com', 'user@example.com']  # Doit être supprimé
        })
        
        filtered_df, removed = DataMinimizer.filter_dataframe_columns(df, 'TEST')
        
        self.assertIn('Open', filtered_df.columns)
        self.assertIn('Close', filtered_df.columns)
        self.assertNotIn('user_id', filtered_df.columns)
        self.assertNotIn('email', filtered_df.columns)
        self.assertEqual(len(removed), 2)
    
    def test_pii_validation(self):
        """Test de validation PII"""
        # Données valides
        valid_data = {
            'Open': [100, 101],
            'Close': [101, 102]
        }
        is_valid, violations = DataMinimizer.validate_no_pii(valid_data)
        self.assertTrue(is_valid)
        
        # Données invalides
        invalid_data = {
            'Open': [100, 101],
            'user_id': [123, 456]
        }
        is_valid, violations = DataMinimizer.validate_no_pii(invalid_data)
        self.assertFalse(is_valid)
    
    def test_retention_policy(self):
        """Test de politique de rétention"""
        self.assertEqual(DataRetentionPolicy.get_retention_period('market_data'), 1)
        self.assertEqual(DataRetentionPolicy.get_retention_period('conversation'), 0)
        self.assertTrue(DataRetentionPolicy.should_retain('market_data'))
        self.assertFalse(DataRetentionPolicy.should_retain('conversation'))
    
    def test_data_minimization(self):
        """Test de minimisation des données"""
        data = {
            'Open': [100, 101],
            'Close': [101, 102],
            'unauthorized_field': 'value',
            'user_name': 'John Doe'
        }
        
        minimized = DataMinimizer.minimize_cache_data(data)
        
        self.assertIn('Open', minimized)
        self.assertIn('Close', minimized)
        self.assertNotIn('unauthorized_field', minimized)
        self.assertNotIn('user_name', minimized)

if __name__ == '__main__':
    unittest.main()
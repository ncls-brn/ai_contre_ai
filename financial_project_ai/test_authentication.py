# test_authentication.py
import unittest
from authentication import (
    PasswordValidator,
    PasswordHasher,
    MFAManager,
    JWTManager,
    AuthenticationManager,
    AuthStatus
)
import time

class TestPasswordValidator(unittest.TestCase):
    
    def test_valid_password(self):
        password = "MyS3cur3P@ssw0rd!"
        is_valid, error = PasswordValidator.validate(password, "testuser")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_too_short(self):
        password = "Short1!"
        is_valid, error = PasswordValidator.validate(password)
        self.assertFalse(is_valid)
        self.assertIn("at least", error.lower())
    
    def test_missing_uppercase(self):
        password = "mypassword123!"
        is_valid, error = PasswordValidator.validate(password)
        self.assertFalse(is_valid)
        self.assertIn("uppercase", error.lower())
    
    def test_common_password(self):
        password = "Password123!"
        is_valid, error = PasswordValidator.validate(password)
        self.assertFalse(is_valid)
        self.assertIn("common", error.lower())
    
    def test_contains_username(self):
        password = "JohnDoe123!"
        is_valid, error = PasswordValidator.validate(password, "johndoe")
        self.assertFalse(is_valid)
        self.assertIn("username", error.lower())
    
    def test_password_strength(self):
        weak = "Pass123!"
        moderate = "MyPassword123!"
        strong = "MyV3ryS3cur3P@ssw0rd!2024"
        
        self.assertLess(PasswordValidator.calculate_strength(weak), 50)
        self.assertGreater(PasswordValidator.calculate_strength(moderate), 50)
        self.assertGreater(PasswordValidator.calculate_strength(strong), 75)


class TestPasswordHasher(unittest.TestCase):
    
    def test_hash_and_verify(self):
        password = "MySecurePassword123!"
        salt = PasswordHasher.generate_salt()
        hash1 = PasswordHasher.hash_password(password, salt)
        
        # Vérifier que le hash est correct
        self.assertTrue(PasswordHasher.verify_password(password, salt, hash1))
        
        # Vérifier qu'un mauvais mot de passe échoue
        self.assertFalse(PasswordHasher.verify_password("WrongPassword", salt, hash1))
    
    def test_different_salts_different_hashes(self):
        password = "MySecurePassword123!"
        salt1 = PasswordHasher.generate_salt()
        salt2 = PasswordHasher.generate_salt()
        
        hash1 = PasswordHasher.hash_password(password, salt1)
        hash2 = PasswordHasher.hash_password(password, salt2)
        
        self.assertNotEqual(hash1, hash2)


class TestMFAManager(unittest.TestCase):
    
    def test_generate_secret(self):
        secret = MFAManager.generate_secret()
        self.assertEqual(len(secret), 32)  # Base32 secret
    
    def test_verify_token(self):
        import pyotp
        secret = MFAManager.generate_secret()
        totp = pyotp.TOTP(secret)
        
        # Générer un token valide
        valid_token = totp.now()
        
        # Vérifier le token
        self.assertTrue(MFAManager.verify_token(secret, valid_token))
        
        # Vérifier un token invalide
        self.assertFalse(MFAManager.verify_token(secret, "000000"))
    
    def test_qr_code_generation(self):
        secret = MFAManager.generate_secret()
        qr_code = MFAManager.generate_qr_code(secret, "testuser")
        
        # Vérifier que c'est une image base64
        self.assertTrue(qr_code.startswith("data:image/png;base64,"))


class TestJWTManager(unittest.TestCase):
    
    def setUp(self):
        self.jwt_manager = JWTManager(secret_key="test_secret_key_12345")
    
    def test_create_and_verify_access_token(self):
        username = "testuser"
        token = self.jwt_manager.create_access_token(username)
        
        is_valid, payload = self.jwt_manager.verify_token(token, token_type='access')
        
        self.assertTrue(is_valid)
        self.assertEqual(payload['sub'], username)
        self.assertEqual(payload['type'], 'access')
    
    def test_create_and_verify_refresh_token(self):
        username = "testuser"
        token = self.jwt_manager.create_refresh_token(username)
        
        is_valid, payload = self.jwt_manager.verify_token(token, token_type='refresh')
        
        self.assertTrue(is_valid)
        self.assertEqual(payload['sub'], username)
        self.assertEqual(payload['type'], 'refresh')
    
    def test_wrong_token_type(self):
        username = "testuser"
        access_token = self.jwt_manager.create_access_token(username)
        
        # Essayer de vérifier comme refresh token
        is_valid, _ = self.jwt_manager.verify_token(access_token, token_type='refresh')
        self.assertFalse(is_valid)
    
    def test_refresh_access_token(self):
        username = "testuser"
        refresh_token = self.jwt_manager.create_refresh_token(username)
        
        new_access_token = self.jwt_manager.refresh_access_token(refresh_token)
        
        self.assertIsNotNone(new_access_token)
        
        is_valid, payload = self.jwt_manager.verify_token(new_access_token, token_type='access')
        self.assertTrue(is_valid)
        self.assertEqual(payload['sub'], username)


class TestAuthenticationManager(unittest.TestCase):
    
    def setUp(self):
        self.auth = AuthenticationManager(jwt_secret="test_secret", storage_path=".test_auth")
    
    def tearDown(self):
        # Nettoyer les fichiers de test
        import shutil
        try:
            shutil.rmtree(".test_auth")
            shutil.rmtree(".sessions")
        except:
            pass
    
    def test_register_user(self):
        status, message = self.auth.register_user("testuser", "MySecureP@ss123!")
        self.assertEqual(status, AuthStatus.SUCCESS)
        self.assertIn("testuser", self.auth.users)
    
    def test_register_weak_password(self):
        status, message = self.auth.register_user("testuser", "weak")
        self.assertEqual(status, AuthStatus.WEAK_PASSWORD)
    
    def test_login_success(self):
        # Enregistrer un utilisateur
        self.auth.register_user("testuser", "MySecureP@ss123!")
        
        # Se connecter
        status, result = self.auth.login("testuser", "MySecureP@ss123!")
        
        self.assertEqual(status, AuthStatus.SUCCESS)
        self.assertIn('access_token', result)
        self.assertIn('refresh_token', result)
        self.assertIn('session_id', result)
    
    def test_login_wrong_password(self):
        self.auth.register_user("testuser", "MySecureP@ss123!")
        
        status, result = self.auth.login("testuser", "WrongPassword")
        
        self.assertEqual(status, AuthStatus.INVALID_CREDENTIALS)
    
    def test_account_lockout(self):
        self.auth.register_user("testuser", "MySecureP@ss123!")
        
        # 5 tentatives échouées
        for _ in range(5):
            self.auth.login("testuser", "WrongPassword")
        
        # 6ème tentative devrait verrouiller le compte
        status, result = self.auth.login("testuser", "MySecureP@ss123!")
        
        self.assertEqual(status, AuthStatus.ACCOUNT_LOCKED)
    
    def test_mfa_flow(self):
        # Enregistrer et activer MFA
        self.auth.register_user("testuser", "MySecureP@ss123!")
        success, secret, qr = self.auth.enable_mfa("testuser")
        
        self.assertTrue(success)
        self.assertTrue(len(secret) > 0)
        
        # Tenter de se connecter sans MFA
        status, result = self.auth.login("testuser", "MySecureP@ss123!")
        self.assertEqual(status, AuthStatus.MFA_REQUIRED)
        
        # Générer un token MFA valide
        import pyotp
        totp = pyotp.TOTP(secret)
        valid_token = totp.now()
        
        # Se connecter avec MFA
        status, result = self.auth.login("testuser", "MySecureP@ss123!", valid_token)
        self.assertEqual(status, AuthStatus.SUCCESS)


if __name__ == '__main__':
    unittest.main()
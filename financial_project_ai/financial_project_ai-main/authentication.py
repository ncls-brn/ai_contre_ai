# authentication.py
import jwt
import pyotp
import qrcode
import hashlib
import secrets
import re
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
import logging
from cryptography.fernet import Fernet
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuthStatus(Enum):
    """Statuts d'authentification"""
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    MFA_REQUIRED = "mfa_required"
    MFA_INVALID = "mfa_invalid"
    ACCOUNT_LOCKED = "account_locked"
    WEAK_PASSWORD = "weak_password"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"


@dataclass
class User:
    """Modèle utilisateur"""
    username: str
    password_hash: str
    salt: str
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PasswordValidator:
    """Validateur de mots de passe robustes"""
    
    MIN_LENGTH = 12
    MAX_LENGTH = 128
    
    # Patterns pour vérifier la complexité
    PATTERNS = {
        'uppercase': r'[A-Z]',
        'lowercase': r'[a-z]',
        'digit': r'\d',
        'special': r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]'
    }
    
    # Liste de mots de passe courants à interdire
    COMMON_PASSWORDS = {
        'password', 'password123', '123456', '12345678', 'qwerty',
        'abc123', 'monkey', '1234567', 'letmein', 'trustno1',
        'dragon', 'baseball', 'iloveyou', 'master', 'sunshine',
        'ashley', 'bailey', 'passw0rd', 'shadow', '123123'
    }
    
    @staticmethod
    def validate(password: str, username: str = "") -> Tuple[bool, str]:
        """
        Valider la robustesse d'un mot de passe
        
        Args:
            password: Mot de passe à valider
            username: Nom d'utilisateur (pour éviter qu'il soit dans le mot de passe)
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Vérifier la longueur
        if len(password) < PasswordValidator.MIN_LENGTH:
            return False, f"Password must be at least {PasswordValidator.MIN_LENGTH} characters"
        
        if len(password) > PasswordValidator.MAX_LENGTH:
            return False, f"Password must not exceed {PasswordValidator.MAX_LENGTH} characters"
        
        # Vérifier la complexité
        missing = []
        for name, pattern in PasswordValidator.PATTERNS.items():
            if not re.search(pattern, password):
                missing.append(name)
        
        if missing:
            return False, f"Password must contain: {', '.join(missing)}"
        
        # Vérifier contre les mots de passe courants
        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            return False, "Password is too common"
        
        # Vérifier que le mot de passe ne contient pas le nom d'utilisateur
        if username and username.lower() in password.lower():
            return False, "Password cannot contain username"
        
        # Vérifier les séquences répétées
        if re.search(r'(.)\1{2,}', password):
            return False, "Password cannot contain repeated characters"
        
        # Vérifier les séquences communes (123, abc)
        sequences = ['123', '234', '345', '456', '567', '678', '789', 
                    'abc', 'bcd', 'cde', 'def', 'efg', 'fgh']
        for seq in sequences:
            if seq in password.lower():
                return False, "Password cannot contain common sequences"
        
        return True, ""
    
    @staticmethod
    def calculate_strength(password: str) -> int:
        """
        Calculer la force d'un mot de passe (0-100)
        
        Args:
            password: Mot de passe à évaluer
        
        Returns:
            int: Score de force (0-100)
        """
        score = 0
        
        # Longueur
        if len(password) >= 12:
            score += 20
        if len(password) >= 16:
            score += 10
        if len(password) >= 20:
            score += 10
        
        # Complexité
        for pattern in PasswordValidator.PATTERNS.values():
            if re.search(pattern, password):
                score += 10
        
        # Diversité des caractères
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 20)
        
        # Entropie
        entropy = len(set(password)) / len(password) if password else 0
        score += int(entropy * 20)
        
        return min(score, 100)


class PasswordHasher:
    """Gestionnaire de hachage de mots de passe"""
    
    ITERATIONS = 100000  # Nombre d'itérations PBKDF2
    
    @staticmethod
    def generate_salt() -> str:
        """Générer un salt aléatoire"""
        return secrets.token_hex(32)
    
    @staticmethod
    def hash_password(password: str, salt: str) -> str:
        """
        Hasher un mot de passe avec PBKDF2
        
        Args:
            password: Mot de passe en clair
            salt: Salt
        
        Returns:
            str: Hash du mot de passe
        """
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt_bytes,
            PasswordHasher.ITERATIONS
        )
        
        return hash_bytes.hex()
    
    @staticmethod
    def verify_password(password: str, salt: str, password_hash: str) -> bool:
        """
        Vérifier un mot de passe
        
        Args:
            password: Mot de passe à vérifier
            salt: Salt
            password_hash: Hash attendu
        
        Returns:
            bool: True si le mot de passe est correct
        """
        computed_hash = PasswordHasher.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, password_hash)


class MFAManager:
    """Gestionnaire d'authentification multi-facteurs (TOTP)"""
    
    @staticmethod
    def generate_secret() -> str:
        """Générer un secret TOTP"""
        return pyotp.random_base32()
    
    @staticmethod
    def get_provisioning_uri(secret: str, username: str, issuer: str = "FinancialApp") -> str:
        """
        Obtenir l'URI de provisioning pour QR code
        
        Args:
            secret: Secret TOTP
            username: Nom d'utilisateur
            issuer: Nom de l'application
        
        Returns:
            str: URI de provisioning
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=issuer)
    
    @staticmethod
    def generate_qr_code(secret: str, username: str) -> str:
        """
        Générer un QR code en base64
        
        Args:
            secret: Secret TOTP
            username: Nom d'utilisateur
        
        Returns:
            str: Image QR code en base64
        """
        uri = MFAManager.get_provisioning_uri(secret, username)
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convertir en base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def verify_token(secret: str, token: str) -> bool:
        """
        Vérifier un token TOTP
        
        Args:
            secret: Secret TOTP
            token: Token à vérifier (6 chiffres)
        
        Returns:
            bool: True si le token est valide
        """
        if not token or not token.isdigit() or len(token) != 6:
            return False
        
        totp = pyotp.TOTP(secret)
        # Vérifier avec une fenêtre de ±1 période (30s)
        return totp.verify(token, valid_window=1)


class JWTManager:
    """Gestionnaire de tokens JWT"""
    
    def __init__(self, secret_key: str):
        """
        Initialiser le gestionnaire JWT
        
        Args:
            secret_key: Clé secrète pour signer les tokens
        """
        self.secret_key = secret_key
        self.algorithm = 'HS256'
        self.access_token_expire = timedelta(minutes=15)
        self.refresh_token_expire = timedelta(days=7)
    
    def create_access_token(self, username: str, data: Dict[str, Any] = None) -> str:
        """
        Créer un token d'accès JWT
        
        Args:
            username: Nom d'utilisateur
            data: Données supplémentaires à inclure
        
        Returns:
            str: Token JWT
        """
        if data is None:
            data = {}
        
        payload = {
            'sub': username,
            'type': 'access',
            'exp': datetime.utcnow() + self.access_token_expire,
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16),  # JWT ID unique
            **data
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(self, username: str) -> str:
        """
        Créer un refresh token JWT
        
        Args:
            username: Nom d'utilisateur
        
        Returns:
            str: Refresh token JWT
        """
        payload = {
            'sub': username,
            'type': 'refresh',
            'exp': datetime.utcnow() + self.refresh_token_expire,
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str, token_type: str = 'access') -> Tuple[bool, Optional[Dict]]:
        """
        Vérifier un token JWT
        
        Args:
            token: Token à vérifier
            token_type: Type de token attendu ('access' ou 'refresh')
        
        Returns:
            Tuple[bool, Optional[Dict]]: (is_valid, payload)
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    'verify_exp': True,
                    'verify_iat': True,
                    'require': ['sub', 'type', 'exp', 'iat', 'jti']
                }
            )
            
            # Vérifier le type de token
            if payload.get('type') != token_type:
                logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
                return False, None
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return False, None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return False, None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Rafraîchir un token d'accès avec un refresh token
        
        Args:
            refresh_token: Refresh token
        
        Returns:
            Optional[str]: Nouveau token d'accès ou None
        """
        is_valid, payload = self.verify_token(refresh_token, token_type='refresh')
        
        if not is_valid or not payload:
            return None
        
        username = payload.get('sub')
        return self.create_access_token(username)


class SessionManager:
    """Gestionnaire de sessions"""
    
    def __init__(self, storage_path: str = ".sessions"):
        """
        Initialiser le gestionnaire de sessions
        
        Args:
            storage_path: Chemin de stockage des sessions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, mode=0o700)
        
        # Dictionnaire en mémoire pour les sessions actives
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, username: str, token_jti: str, data: Dict[str, Any] = None) -> str:
        """
        Créer une nouvelle session
        
        Args:
            username: Nom d'utilisateur
            token_jti: JWT ID du token
            data: Données de session supplémentaires
        
        Returns:
            str: ID de session
        """
        session_id = secrets.token_hex(32)
        
        session_data = {
            'session_id': session_id,
            'username': username,
            'token_jti': token_jti,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'data': data or {}
        }
        
        self.active_sessions[session_id] = session_data
        
        logger.info(f"Session created for user {username}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupérer une session
        
        Args:
            session_id: ID de session
        
        Returns:
            Optional[Dict]: Données de session ou None
        """
        return self.active_sessions.get(session_id)
    
    def update_activity(self, session_id: str):
        """
        Mettre à jour l'activité d'une session
        
        Args:
            session_id: ID de session
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.now().isoformat()
    
    def revoke_session(self, session_id: str):
        """
        Révoquer une session
        
        Args:
            session_id: ID de session
        """
        if session_id in self.active_sessions:
            username = self.active_sessions[session_id].get('username')
            del self.active_sessions[session_id]
            logger.info(f"Session revoked for user {username}")
    
    def revoke_user_sessions(self, username: str):
        """
        Révoquer toutes les sessions d'un utilisateur
        
        Args:
            username: Nom d'utilisateur
        """
        sessions_to_revoke = [
            sid for sid, session in self.active_sessions.items()
            if session.get('username') == username
        ]
        
        for sid in sessions_to_revoke:
            del self.active_sessions[sid]
        
        logger.info(f"Revoked {len(sessions_to_revoke)} sessions for user {username}")
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """
        Nettoyer les sessions expirées
        
        Args:
            max_age_hours: Âge maximum en heures
        """
        now = datetime.now()
        max_age = timedelta(hours=max_age_hours)
        
        expired_sessions = []
        
        for sid, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session['last_activity'])
            if now - last_activity > max_age:
                expired_sessions.append(sid)
        
        for sid in expired_sessions:
            del self.active_sessions[sid]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class AuthenticationManager:
    """Gestionnaire principal d'authentification"""
    
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=30)
    
    def __init__(self, jwt_secret: str, storage_path: str = ".auth"):
        """
        Initialiser le gestionnaire d'authentification
        
        Args:
            jwt_secret: Clé secrète JWT
            storage_path: Chemin de stockage des données
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, mode=0o700)
        
        self.jwt_manager = JWTManager(jwt_secret)
        self.session_manager = SessionManager()
        
        # Base de données utilisateurs (en mémoire pour la démo)
        # En production, utiliser une vraie base de données
        self.users: Dict[str, User] = {}
        
        self._load_users()
    
    def _load_users(self):
        """Charger les utilisateurs depuis le stockage"""
        users_file = self.storage_path / "users.json"
        
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    data = json.load(f)
                
                for username, user_data in data.items():
                    # Reconstruire les objets datetime
                    if user_data.get('locked_until'):
                        user_data['locked_until'] = datetime.fromisoformat(user_data['locked_until'])
                    if user_data.get('created_at'):
                        user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                    if user_data.get('last_login'):
                        user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                    
                    self.users[username] = User(**user_data)
                
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
    
    def _save_users(self):
        """Sauvegarder les utilisateurs"""
        users_file = self.storage_path / "users.json"
        
        try:
            data = {}
            for username, user in self.users.items():
                user_data = {
                    'username': user.username,
                    'password_hash': user.password_hash,
                    'salt': user.salt,
                    'mfa_secret': user.mfa_secret,
                    'mfa_enabled': user.mfa_enabled,
                    'failed_attempts': user.failed_attempts,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                }
                data[username] = user_data
            
            with open(users_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Permissions restrictives
            users_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def register_user(self, username: str, password: str) -> Tuple[AuthStatus, str]:
        """
        Enregistrer un nouvel utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
        
        Returns:
            Tuple[AuthStatus, str]: (status, message)
        """
        # Valider le nom d'utilisateur
        if not username or len(username) < 3:
            return AuthStatus.INVALID_CREDENTIALS, "Username must be at least 3 characters"
        
        if username in self.users:
            return AuthStatus.INVALID_CREDENTIALS, "Username already exists"
        
        # Valider le mot de passe
        is_valid, error = PasswordValidator.validate(password, username)
        if not is_valid:
            return AuthStatus.WEAK_PASSWORD, error
        
        # Hasher le mot de passe
        salt = PasswordHasher.generate_salt()
        password_hash = PasswordHasher.hash_password(password, salt)
        
        # Créer l'utilisateur
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt
        )
        
        self.users[username] = user
        self._save_users()
        
        logger.info(f"User registered: {username}")
        return AuthStatus.SUCCESS, "User registered successfully"
    
    def enable_mfa(self, username: str) -> Tuple[bool, str, str]:
        """
        Activer MFA pour un utilisateur
        
        Args:
            username: Nom d'utilisateur
        
        Returns:
            Tuple[bool, str, str]: (success, secret, qr_code_base64)
        """
        if username not in self.users:
            return False, "", ""
        
        user = self.users[username]
        
        # Générer le secret MFA
        secret = MFAManager.generate_secret()
        qr_code = MFAManager.generate_qr_code(secret, username)
        
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        self._save_users()
        
        logger.info(f"MFA enabled for user: {username}")
        return True, secret, qr_code
    
    def login(self, username: str, password: str, mfa_token: Optional[str] = None) -> Tuple[AuthStatus, Optional[Dict]]:
        """
        Authentifier un utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            mfa_token: Token MFA (si MFA activé)
        
        Returns:
            Tuple[AuthStatus, Optional[Dict]]: (status, tokens_and_session)
        """
        # Vérifier si l'utilisateur existe
        if username not in self.users:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return AuthStatus.INVALID_CREDENTIALS, None
        
        user = self.users[username]
        
        # Vérifier si le compte est verrouillé
        if user.locked_until and datetime.now() < user.locked_until:
            remaining = (user.locked_until - datetime.now()).seconds // 60
            logger.warning(f"Login attempt for locked account: {username}")
            return AuthStatus.ACCOUNT_LOCKED, {'minutes_remaining': remaining}
        
        # Débloquer le compte si la période est passée
        if user.locked_until and datetime.now() >= user.locked_until:
            user.locked_until = None
            user.failed_attempts = 0
        
        # Vérifier le mot de passe
        if not PasswordHasher.verify_password(password, user.salt, user.password_hash):
            user.failed_attempts += 1
            
            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.now() + self.LOCKOUT_DURATION
                self._save_users()
                logger.warning(f"Account locked due to failed attempts: {username}")
                return AuthStatus.ACCOUNT_LOCKED, {'minutes_remaining': self.LOCKOUT_DURATION.seconds // 60}
            
            self._save_users()
            logger.warning(f"Failed login attempt for user: {username}")
            return AuthStatus.INVALID_CREDENTIALS, None
        
        # Vérifier MFA si activé
        if user.mfa_enabled:
            if not mfa_token:
                return AuthStatus.MFA_REQUIRED, None
            
            if not MFAManager.verify_token(user.mfa_secret, mfa_token):
                logger.warning(f"Invalid MFA token for user: {username}")
                return AuthStatus.MFA_INVALID, None
        
        # Authentification réussie
        user.failed_attempts = 0
        user.last_login = datetime.now()
        self._save_users()
        
        # Créer les tokens JWT
        access_token = self.jwt_manager.create_access_token(username)
        refresh_token = self.jwt_manager.create_refresh_token(username)
        
        # Décoder pour obtenir le JTI
        _, payload = self.jwt_manager.verify_token(access_token)
        jti = payload['jti']
        
        # Créer la session
        session_id = self.session_manager.create_session(username, jti)
        
        logger.info(f"Successful login for user: {username}")
        
        return AuthStatus.SUCCESS, {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'session_id': session_id,
            'username': username
        }
    
    def verify_session(self, access_token: str, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Vérifier une session
        
        Args:
            access_token: Token d'accès JWT
            session_id: ID de session
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, username)
        """
        # Vérifier le token JWT
        is_valid, payload = self.jwt_manager.verify_token(access_token)
        if not is_valid or not payload:
            return False, None
        
        username = payload['sub']
        jti = payload['jti']
        
        # Vérifier la session
        session = self.session_manager.get_session(session_id)
        if not session:
            return False, None
        
        # Vérifier que le token correspond à la session
        if session['token_jti'] != jti or session['username'] != username:
            return False, None
        
        # Mettre à jour l'activité
        self.session_manager.update_activity(session_id)
        
        return True, username
    
    
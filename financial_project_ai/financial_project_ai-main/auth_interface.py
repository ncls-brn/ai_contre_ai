# auth_interface.py
import gradio as gr
from authentication import (
    AuthenticationManager, 
    AuthStatus, 
    PasswordValidator
)
import os
import secrets
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Initialiser le gestionnaire d'authentification
JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_hex(32))
auth_manager = AuthenticationManager(jwt_secret=JWT_SECRET)

# État de session global (en production, utiliser Redis ou similaire)
current_session = {
    'authenticated': False,
    'username': None,
    'session_id': None,
    'access_token': None,
    'refresh_token': None
}


def register_user(username: str, password: str, confirm_password: str) -> Tuple[str, str, bool]:
    """
    Enregistrer un nouvel utilisateur
    
    Returns:
        Tuple[str, str, bool]: (message, password_strength, success)
    """
    try:
        # Vérifier que les mots de passe correspondent
        if password != confirm_password:
            return "❌ Passwords do not match", "", False
        
        # Calculer la force du mot de passe
        strength = PasswordValidator.calculate_strength(password)
        strength_label = f"Password strength: {strength}/100"
        
        if strength < 50:
            strength_label += " (Too weak)"
        elif strength < 75:
            strength_label += " (Moderate)"
        else:
            strength_label += " (Strong)"
        
        # Enregistrer l'utilisateur
        status, message = auth_manager.register_user(username, password)
        
        if status == AuthStatus.SUCCESS:
            logger.info(f"User registered successfully: {username}")
            return f"✅ {message}", strength_label, True
        elif status == AuthStatus.WEAK_PASSWORD:
            return f"❌ {message}\n\nRequirements:\n- At least 12 characters\n- Uppercase letters\n- Lowercase letters\n- Numbers\n- Special characters", strength_label, False
        else:
            return f"❌ {message}", strength_label, False
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return f"❌ Error: {str(e)}", "", False


def setup_mfa(username: str, password: str) -> Tuple[str, str, bool]:
    """
    Configurer MFA pour un utilisateur
    
    Returns:
        Tuple[str, str, bool]: (message, qr_code, success)
    """
    try:
        # Vérifier les identifiants
        status, result = auth_manager.login(username, password)
        
        if status != AuthStatus.SUCCESS and status != AuthStatus.MFA_REQUIRED:
            return "❌ Invalid credentials", "", False
        
        # Activer MFA
        success, secret, qr_code = auth_manager.enable_mfa(username)
        
        if success:
            message = f"""✅ MFA enabled successfully!

**Important:**
1. Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)
2. Save your backup code: `{secret}`
3. You will need to enter a 6-digit code from your app each time you log in

**Keep your backup code safe!** If you lose access to your authenticator app, you'll need this code to regain access.
"""
            logger.info(f"MFA enabled for user: {username}")
            return message, qr_code, True
        else:
            return "❌ Failed to enable MFA", "", False
            
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        return f"❌ Error: {str(e)}", "", False


def login_user(username: str, password: str, mfa_token: str = "") -> Tuple[str, bool, bool]:
    """
    Authentifier un utilisateur
    
    Returns:
        Tuple[str, bool, bool]: (message, success, mfa_required)
    """
    global current_session
    
    try:
        # Tentative de connexion
        status, result = auth_manager.login(username, password, mfa_token or None)
        
        if status == AuthStatus.SUCCESS:
            # Sauvegarder la session
            current_session = {
                'authenticated': True,
                'username': result['username'],
                'session_id': result['session_id'],
                'access_token': result['access_token'],
                'refresh_token': result['refresh_token']
            }
            
            logger.info(f"User logged in: {username}")
            return f"✅ Welcome back, {username}!", True, False
        
        elif status == AuthStatus.MFA_REQUIRED:
            return "🔐 MFA code required. Please enter your 6-digit code from your authenticator app.", False, True
        
        elif status == AuthStatus.MFA_INVALID:
            return "❌ Invalid MFA code. Please try again.", False, True
        
        elif status == AuthStatus.ACCOUNT_LOCKED:
            minutes = result.get('minutes_remaining', 30)
            return f"🔒 Account locked due to too many failed attempts. Try again in {minutes} minutes.", False, False
        
        else:
            return "❌ Invalid username or password", False, False
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return f"❌ Error: {str(e)}", False, False


def logout_user() -> str:
    """Déconnecter l'utilisateur"""
    global current_session
    
    if current_session['authenticated']:
        session_id = current_session['session_id']
        username = current_session['username']
        
        auth_manager.logout(session_id)
        
        current_session = {
            'authenticated': False,
            'username': None,
            'session_id': None,
            'access_token': None,
            'refresh_token': None
        }
        
        logger.info(f"User logged out: {username}")
        return f"✅ Logged out successfully"
    
    return "ℹ️ Not logged in"


def check_session() -> Tuple[bool, str]:
    """
    Vérifier si la session est valide
    
    Returns:
        Tuple[bool, str]: (is_valid, username)
    """
    global current_session
    
    if not current_session['authenticated']:
        return False, ""
    
    is_valid, username = auth_manager.verify_session(
        current_session['access_token'],
        current_session['session_id']
    )
    
    if not is_valid:
        # Session invalide, déconnecter
        current_session = {
            'authenticated': False,
            'username': None,
            'session_id': None,
            'access_token': None,
            'refresh_token': None
        }
        return False, ""
    
    return True, username


def get_password_requirements() -> str:
    """Obtenir les exigences de mot de passe"""
    return """**Password Requirements:**

✓ At least 12 characters
✓ At least one uppercase letter (A-Z)
✓ At least one lowercase letter (a-z)
✓ At least one number (0-9)
✓ At least one special character (!@#$%^&*()_+-=[]{}...)
✗ Cannot contain username
✗ Cannot be a common password
✗ Cannot contain repeated characters (aaa, 111)
✗ Cannot contain sequences (123, abc)
"""


def create_auth_interface() -> gr.Blocks:
    """Créer l'interface d'authentification Gradio"""
    
    with gr.Blocks(title="🔐 Secure Authentication") as auth_ui:
        gr.Markdown("# 🔐 Secure Authentication System")
        gr.Markdown("*Multi-Factor Authentication • JWT Sessions • Strong Password Policy*")
        
        with gr.Tabs() as tabs:
            # Onglet Connexion
            with gr.Tab("🔓 Login"):
                gr.Markdown("### Sign in to your account")
                
                with gr.Column():
                    login_username = gr.Textbox(
                        label="Username",
                        placeholder="Enter your username",
                        max_lines=1
                    )
                    login_password = gr.Textbox(
                        label="Password",
                        placeholder="Enter your password",
                        type="password",
                        max_lines=1
                    )
                    login_mfa = gr.Textbox(
                        label="MFA Code (if enabled)",
                        placeholder="6-digit code from authenticator app",
                        max_lines=1,
                        visible=False
                    )
                    
                    login_button = gr.Button("🔓 Sign In", variant="primary")
                    login_message = gr.Markdown("")
            
            # Onglet Inscription
            with gr.Tab("📝 Register"):
                gr.Markdown("### Create a new account")
                
                with gr.Column():
                    reg_username = gr.Textbox(
                        label="Username",
                        placeholder="Choose a username (min 3 characters)",
                        max_lines=1
                    )
                    reg_password = gr.Textbox(
                        label="Password",
                        placeholder="Enter a strong password",
                        type="password",
                        max_lines=1
                    )
                    reg_confirm = gr.Textbox(
                        label="Confirm Password",
                        placeholder="Re-enter your password",
                        type="password",
                        max_lines=1
                    )
                    
                    password_strength = gr.Textbox(
                        label="Password Strength",
                        interactive=False
                    )
                    
                    with gr.Accordion("📋 Password Requirements", open=False):
                        gr.Markdown(get_password_requirements())
                    
                    register_button = gr.Button("📝 Create Account", variant="primary")
                    register_message = gr.Markdown("")
            
            # Onglet MFA
            with gr.Tab("🔐 Setup MFA"):
                gr.Markdown("### Enable Two-Factor Authentication")
                gr.Markdown("Secure your account with an authenticator app (Google Authenticator, Authy, Microsoft Authenticator, etc.)")
                
                with gr.Column():
                    mfa_username = gr.Textbox(
                        label="Username",
                        placeholder="Enter your username",
                        max_lines=1
                    )
                    mfa_password = gr.Textbox(
                        label="Password",
                        placeholder="Enter your password",
                        type="password",
                        max_lines=1
                    )
                    
                    mfa_button = gr.Button("🔐 Enable MFA", variant="primary")
                    mfa_qr = gr.Image(label="QR Code - Scan with your authenticator app")
                    mfa_message = gr.Markdown("")
            
            # Onglet Session
            with gr.Tab("👤 Session"):
                gr.Markdown("### Current Session")
                
                session_info = gr.Markdown("ℹ️ Not logged in")
                refresh_button = gr.Button("🔄 Check Session")
                logout_button = gr.Button("🚪 Logout", variant="secondary")
        
        # Event handlers
        def handle_login(username, password, mfa_token):
            message, success, mfa_required = login_user(username, password, mfa_token)
            
            # Afficher/masquer le champ MFA
            mfa_visible = mfa_required
            
            if success:
                session_text = f"✅ **Logged in as:** {current_session['username']}\n\n**Session ID:** `{current_session['session_id'][:16]}...`\n\n**Token expires in:** 15 minutes"
                return message, gr.update(visible=False), session_text
            else:
                return message, gr.update(visible=mfa_visible), "ℹ️ Not logged in"
        
        def handle_register(username, password, confirm):
            message, strength, success = register_user(username, password, confirm)
            return message, strength
        
        def handle_mfa_setup(username, password):
            message, qr_code, success = setup_mfa(username, password)
            return message, qr_code if success else None
        
        def handle_logout():
            message = logout_user()
            return message, "ℹ️ Not logged in"
        
        def handle_refresh():
            is_valid, username = check_session()
            if is_valid:
                return f"✅ **Logged in as:** {username}\n\n**Session ID:** `{current_session['session_id'][:16]}...`\n\n**Token expires in:** 15 minutes"
            else:
                return "ℹ️ Session expired or not logged in"
        
        # Connecter les événements
        login_button.click(
            fn=handle_login,
            inputs=[login_username, login_password, login_mfa],
            outputs=[login_message, login_mfa, session_info]
        )
        
        register_button.click(
            fn=handle_register,
            inputs=[reg_username, reg_password, reg_confirm],
            outputs=[register_message, password_strength]
        )
        
        mfa_button.click(
            fn=handle_mfa_setup,
            inputs=[mfa_username, mfa_password],
            outputs=[mfa_message, mfa_qr]
        )
        
        logout_button.click(
            fn=handle_logout,
            outputs=[login_message, session_info]
        )
        
        refresh_button.click(
            fn=handle_refresh,
            outputs=[session_info]
        )
    
    return auth_ui


# Middleware pour protéger l'application principale
def require_authentication(func):
    """Décorateur pour exiger l'authentification"""
    def wrapper(*args, **kwargs):
        is_valid, username = check_session()
        
        if not is_valid:
            return "❌ Authentication required. Please log in first.", None, None, None
        
        return func(*args, **kwargs)
    
    return wrapper
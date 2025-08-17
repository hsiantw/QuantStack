import streamlit as st
import hashlib
import sqlite3
import os
from datetime import datetime, timedelta
import json
from typing import Dict, Optional, Any

class AuthManager:
    """Authentication and user management system"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                subscription_tier TEXT DEFAULT 'free'
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                preference_key TEXT NOT NULL,
                preference_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, preference_key)
            )
        ''')
        
        # User strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                strategy_config TEXT,
                backtest_results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_favorite BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User portfolios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                portfolio_name TEXT NOT NULL,
                portfolio_data TEXT,
                portfolio_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """Create a new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username or email already exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return {"success": False, "message": "Username or email already exists"}
            
            # Create user
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Set default preferences
            self.set_default_preferences(user_id)
            
            return {"success": True, "message": "Account created successfully", "user_id": user_id}
            
        except Exception as e:
            return {"success": False, "message": f"Error creating account: {str(e)}"}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            cursor.execute('''
                SELECT id, username, email, full_name, subscription_tier 
                FROM users 
                WHERE (username = ? OR email = ?) AND password_hash = ? AND is_active = 1
            ''', (username, username, password_hash))
            
            user = cursor.fetchone()
            
            if user:
                user_id = user[0]
                
                # Update last login
                cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
                conn.commit()
                
                # Create session token
                session_token = self.create_session(user_id)
                
                user_data = {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "full_name": user[3],
                    "subscription_tier": user[4],
                    "session_token": session_token
                }
                
                conn.close()
                return {"success": True, "user": user_data}
            else:
                conn.close()
                return {"success": False, "message": "Invalid username/email or password"}
                
        except Exception as e:
            return {"success": False, "message": f"Authentication error: {str(e)}"}
    
    def create_session(self, user_id: int) -> str:
        """Create a session token for the user"""
        import secrets
        
        session_token = secrets.token_hex(32)
        expires_at = datetime.now() + timedelta(days=30)  # 30-day session
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean up expired sessions
        cursor.execute("DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP")
        
        # Create new session
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, session_token, expires_at))
        
        conn.commit()
        conn.close()
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.full_name, u.subscription_tier
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
            ''', (session_token,))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "full_name": user[3],
                    "subscription_tier": user[4]
                }
            return None
            
        except Exception:
            return None
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
            conn.commit()
            conn.close()
            
            return True
        except Exception:
            return False
    
    def set_default_preferences(self, user_id: int):
        """Set default user preferences"""
        default_prefs = {
            "theme": "dark",
            "default_portfolio_size": "100000",
            "risk_tolerance": "moderate",
            "preferred_strategies": "[]",
            "notification_settings": '{"email": true, "trading_signals": true}',
            "dashboard_layout": "standard"
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in default_prefs.items():
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value)
                VALUES (?, ?, ?)
            ''', (user_id, key, value))
        
        conn.commit()
        conn.close()
    
    def get_user_preference(self, user_id: int, key: str, default: Any = None) -> Any:
        """Get user preference value"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT preference_value FROM user_preferences 
                WHERE user_id = ? AND preference_key = ?
            ''', (user_id, key))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                value = result[0]
                # Try to parse JSON if it looks like JSON
                if value.startswith('{') or value.startswith('['):
                    try:
                        return json.loads(value)
                    except:
                        return value
                return value
            return default
            
        except Exception:
            return default
    
    def set_user_preference(self, user_id: int, key: str, value: Any) -> bool:
        """Set user preference value"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert to JSON if it's a dict or list
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, value))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def save_user_strategy(self, user_id: int, strategy_name: str, strategy_type: str, 
                          strategy_config: Dict, backtest_results: Dict = None) -> bool:
        """Save user's trading strategy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            config_json = json.dumps(strategy_config)
            results_json = json.dumps(backtest_results) if backtest_results else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_strategies 
                (user_id, strategy_name, strategy_type, strategy_config, backtest_results, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, strategy_name, strategy_type, config_json, results_json))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def get_user_strategies(self, user_id: int) -> list:
        """Get all user strategies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT strategy_name, strategy_type, strategy_config, backtest_results, 
                       created_at, updated_at, is_favorite
                FROM user_strategies 
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,))
            
            strategies = []
            for row in cursor.fetchall():
                strategy = {
                    "name": row[0],
                    "type": row[1],
                    "config": json.loads(row[2]) if row[2] else {},
                    "backtest_results": json.loads(row[3]) if row[3] else None,
                    "created_at": row[4],
                    "updated_at": row[5],
                    "is_favorite": bool(row[6])
                }
                strategies.append(strategy)
            
            conn.close()
            return strategies
            
        except Exception:
            return []
    
    def save_user_portfolio(self, user_id: int, portfolio_name: str, 
                           portfolio_data: Dict, portfolio_metrics: Dict = None) -> bool:
        """Save user's portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data_json = json.dumps(portfolio_data)
            metrics_json = json.dumps(portfolio_metrics) if portfolio_metrics else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_portfolios 
                (user_id, portfolio_name, portfolio_data, portfolio_metrics, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, portfolio_name, data_json, metrics_json))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def get_user_portfolios(self, user_id: int) -> list:
        """Get all user portfolios"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT portfolio_name, portfolio_data, portfolio_metrics, created_at, updated_at
                FROM user_portfolios 
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,))
            
            portfolios = []
            for row in cursor.fetchall():
                portfolio = {
                    "name": row[0],
                    "data": json.loads(row[1]) if row[1] else {},
                    "metrics": json.loads(row[2]) if row[2] else None,
                    "created_at": row[3],
                    "updated_at": row[4]
                }
                portfolios.append(portfolio)
            
            conn.close()
            return portfolios
            
        except Exception:
            return []

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()

def show_login_form():
    """Display login form"""
    st.markdown("""
    <div style="max-width: 400px; margin: 0 auto; padding: 2rem; 
                background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); 
                border-radius: 15px; box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);">
        <h2 style="text-align: center; color: #00d4ff; margin-bottom: 2rem;">🔐 Login</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username or Email", placeholder="Enter your username or email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login", type="primary", use_container_width=True)
        with col2:
            forgot_button = st.form_submit_button("Forgot Password?", use_container_width=True)
        
        if login_button and username and password:
            auth_manager = st.session_state.auth_manager
            result = auth_manager.authenticate_user(username, password)
            
            if result["success"]:
                st.session_state.authenticated = True
                st.session_state.user = result["user"]
                st.success("Login successful! Welcome back.")
                st.rerun()
            else:
                st.error(result["message"])
        
        if forgot_button:
            st.info("Password reset functionality will be available soon. Please contact support.")

def show_signup_form():
    """Display signup form"""
    st.markdown("""
    <div style="max-width: 400px; margin: 0 auto; padding: 2rem; 
                background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); 
                border-radius: 15px; box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);">
        <h2 style="text-align: center; color: #00d4ff; margin-bottom: 2rem;">📝 Sign Up</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Choose a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        
        signup_button = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        
        if signup_button:
            if not all([full_name, username, email, password, confirm_password]):
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            elif not terms_accepted:
                st.error("Please accept the Terms of Service and Privacy Policy.")
            else:
                auth_manager = st.session_state.auth_manager
                result = auth_manager.create_user(username, email, password, full_name)
                
                if result["success"]:
                    st.success("Account created successfully! You can now log in.")
                    st.session_state.show_signup = False
                    st.rerun()
                else:
                    st.error(result["message"])

def show_auth_page():
    """Display authentication page with login/signup tabs"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 style="color: #00d4ff; font-size: 3rem; margin-bottom: 1rem;">
            💼 Quantitative Finance Platform
        </h1>
        <p style="color: #888; font-size: 1.2rem; margin-bottom: 3rem;">
            Professional trading strategies, AI-powered analysis, and portfolio optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        show_login_form()
    
    with tab2:
        show_signup_form()
    
    # Features preview
    st.markdown("---")
    st.markdown("### 🌟 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI-Powered Analysis**
        - Machine learning predictions
        - Automated strategy optimization
        - Real-time market intelligence
        """)
    
    with col2:
        st.markdown("""
        **📊 Advanced Strategies**
        - Pairs trading with cointegration
        - Mean reversion algorithms
        - Portfolio optimization (MPT)
        """)
    
    with col3:
        st.markdown("""
        **💼 Professional Tools**
        - Risk management systems
        - Backtesting frameworks
        - Live trading integration
        """)

def show_user_menu():
    """Display user menu in sidebar"""
    if st.session_state.authenticated and st.session_state.user:
        user = st.session_state.user
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 👤 Welcome, {user['username']}")
            st.markdown(f"**Tier:** {user['subscription_tier'].title()}")
            
            # User menu options
            if st.button("⚙️ Account Settings", use_container_width=True):
                st.switch_page("pages/user_settings.py")
            
            if st.button("💾 My Strategies", use_container_width=True):
                st.switch_page("pages/user_settings.py")
            
            if st.button("📊 My Portfolios", use_container_width=True):
                st.switch_page("pages/user_settings.py")
            
            if st.button("🚪 Logout", use_container_width=True):
                auth_manager = st.session_state.auth_manager
                auth_manager.logout_user(user.get("session_token", ""))
                st.session_state.authenticated = False
                st.session_state.user = None
                st.success("Logged out successfully!")
                st.rerun()

def require_auth(func):
    """Decorator to require authentication for functions"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            show_auth_page()
            return None
        return func(*args, **kwargs)
    return wrapper

def save_user_data(key: str, value: Any):
    """Save data associated with current user"""
    if st.session_state.authenticated and st.session_state.user:
        user_id = st.session_state.user['id']
        auth_manager = st.session_state.auth_manager
        return auth_manager.set_user_preference(user_id, key, value)
    return False

def load_user_data(key: str, default: Any = None):
    """Load data associated with current user"""
    if st.session_state.authenticated and st.session_state.user:
        user_id = st.session_state.user['id']
        auth_manager = st.session_state.auth_manager
        return auth_manager.get_user_preference(user_id, key, default)
    return default
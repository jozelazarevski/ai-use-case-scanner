from flask import request, render_template, redirect, url_for, flash, session, g
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from datetime import datetime

# Import the database connection module
import db_connection as db

def login_required(f):
    """Decorator to ensure the user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def init_database():
    """Initialize database - called by app.py"""
    db.initialize_database()

def init_auth_routes(app):
    """Initialize authentication routes"""
    
    @app.before_request
    def load_logged_in_user():
        """Load user data for each request if logged in"""
        user_id = session.get('user_id')
        if user_id is None:
            g.user = None
        else:
            # Convert string UUID to UUID object
            try:
                user_id_uuid = uuid.UUID(user_id)
                g.user = db.get_user_by_id(user_id_uuid)
            except (ValueError, TypeError):
                g.user = None
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration"""
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            
            # Validate input
            error = None
            if not username:
                error = 'Username is required.'
            elif not email:
                error = 'Email is required.'
            elif not password:
                error = 'Password is required.'
            elif db.get_user_by_username(username):
                error = f'User {username} is already registered.'
            
            if error is None:
                # Create new user
                password_hash = generate_password_hash(password)
                user_id = db.create_user(username, email, password_hash)
                
                if user_id:
                    flash('Registration successful! Please log in.', 'success')
                    return redirect(url_for('login'))
                else:
                    error = 'Failed to create user. Username or email may already be in use.'
            
            flash(error, 'error')
        
        return render_template('register.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """User login"""
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            user = db.get_user_by_username(username)
            
            error = None
            if user is None:
                error = 'Incorrect username.'
            elif not check_password_hash(user['password_hash'], password):
                error = 'Incorrect password.'
            
            if error is None:
                # Store user info in session - convert UUID to string for session storage
                session.clear()
                session['user_id'] = str(user['id'])
                session['user_name'] = user['username']
                
                # Update last login time
                db.update_user_last_login(user['id'])
                
                next_page = request.args.get('next', url_for('upload_file'))
                flash(f'Welcome back, {user["username"]}!', 'success')
                return redirect(next_page)
            
            flash(error, 'error')
        
        return render_template('login.html')
    
    @app.route('/logout')
    def logout():
        """User logout"""
        session.clear()
        flash('You have been logged out.', 'success')
        return redirect(url_for('login'))
    
    @app.route('/profile')
    @login_required
    def profile():
        """User profile page"""
        # Convert string UUID from session back to UUID object for database query
        user_id_uuid = uuid.UUID(session['user_id'])
        user = db.get_user_by_id(user_id_uuid)
        return render_template('profile.html', user=user)

    @app.route('/admin')
    @login_required
    def admin_dashboard():
        """Admin dashboard"""
        # Convert string UUID from session back to UUID object for database query
        user_id_uuid = uuid.UUID(session['user_id'])
        user = db.get_user_by_id(user_id_uuid)
        
        # Only allow admin users to access this page
        if not user or user['username'] != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('upload_file'))
        
        # Get database statistics using SQL queries
        with db.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get user count
                cursor.execute("SELECT COUNT(*) AS user_count FROM users")
                user_count = cursor.fetchone()['user_count']
                
                # Get model count
                cursor.execute("SELECT COUNT(*) AS model_count FROM models")
                model_count = cursor.fetchone()['model_count']
                
                # Get use case count
                cursor.execute("SELECT COUNT(*) AS use_case_count FROM use_cases")
                use_case_count = cursor.fetchone()['use_case_count']
                
                # Get embedding count
                cursor.execute("SELECT COUNT(*) AS embedding_count FROM embeddings")
                embedding_count = cursor.fetchone()['embedding_count']
                
                # Get prediction log count
                cursor.execute("SELECT COUNT(*) AS prediction_count FROM prediction_logs")
                prediction_count = cursor.fetchone()['prediction_count']
                
                # Get recent users
                cursor.execute("""
                    SELECT username, email, created_at, last_login 
                    FROM users 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                recent_users = cursor.fetchall()
                
                # Get recent models
                cursor.execute("""
                    SELECT m.name, m.model_type, m.created_at, u.username as owner
                    FROM models m
                    JOIN users u ON m.user_id = u.id
                    ORDER BY m.created_at DESC
                    LIMIT 10
                """)
                recent_models = cursor.fetchall()
        
        # Render the admin dashboard template
        return render_template('admin_dashboard.html', 
                              user_count=user_count,
                              model_count=model_count,
                              use_case_count=use_case_count,
                              embedding_count=embedding_count,
                              prediction_count=prediction_count,
                              recent_users=recent_users,
                              recent_models=recent_models)

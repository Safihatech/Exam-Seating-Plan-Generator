from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils.pdf_export import export_to_pdf
from datetime import datetime
import logging
from functools import wraps

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = 'your_secret_key_change_this_in_production'

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global database instance
db = None

def init_database():
    """Initialize database connection"""
    global db
    try:
        from backend.models.database import Database
        db = Database()
        if db.connect():
            db.create_tables()
            return True
        else:
            print("Failed to connect to database. Using fallback authentication.")
            return False
    except Exception as e:
        print(f"Database initialization error: {e}")
        print("Using fallback authentication.")
        return False

def is_logged_in():
    return session.get('logged_in', False)

def get_user_role():
    return session.get('role', 'student')

def require_login(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not is_logged_in():
                return redirect(url_for('login'))
            if get_user_role() != required_role and get_user_role() != 'admin':
                flash('Access denied. Insufficient permissions.')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def fallback_auth(username, password):
    """Fallback authentication when database is not available"""
    return username == 'admin' and password == 'admin123'

def get_client_ip():
    """Get client IP address"""
    return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

def get_user_agent():
    """Get user agent"""
    return request.headers.get('User-Agent', '')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        # Enforce Administrator-only signup per project requirement
        role = 'admin'
        
        # Validation
        if not all([username, email, password, confirm_password, full_name]):
            flash('All fields are required.')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.')
            return redirect(url_for('signup'))
        
        # Try to create user in database
        if db and db.connection and db.connection.is_connected():
            success, message = db.create_user(username, email, password, full_name, role)
            if success:
                flash('Account created successfully! Please login.')
                return redirect(url_for('login'))
            else:
                flash(message)
                return redirect(url_for('signup'))
        else:
            flash('Database not available. Please contact administrator.')
            return redirect(url_for('signup'))
    
    return render_template('signup/signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Try database authentication first
        if db and db.connection and db.connection.is_connected():
            user = db.authenticate_user(username, password)
            if user:
                if isinstance(user, tuple):
                    user_id = user[0]
                    user_name = user[1] if len(user) > 1 else username
                    email = user[2] if len(user) > 2 else ''
                    full_name = user[4] if len(user) > 4 else ''
                    role = user[5] if len(user) > 5 else ''
                    session['logged_in'] = True
                    session['user_id'] = user_id
                    session['username'] = (user_name or '').strip()
                    session['full_name'] = (full_name or user_name or '').strip()
                    session['role'] = (role or 'student').strip().lower()
                    session['email'] = email
                elif isinstance(user, dict):
                    session['logged_in'] = True
                    session['user_id'] = user.get('id')
                    session['username'] = (user.get('username') or '').strip()
                    session['full_name'] = (user.get('full_name') or user.get('username', '')).strip()
                    session['role'] = (user.get('role') or 'student').strip().lower()
                    session['email'] = user.get('email', '')
                
                # Log successful login
                if db:
                    db.log_activity(session['user_id'], 'LOGIN', f"User {username} logged in", get_client_ip(), get_user_agent())
                
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.')
                return redirect(url_for('login'))
        else:
            # Fallback authentication
            if fallback_auth(username, password):
                session['logged_in'] = True
                session['user_id'] = 1  # Default admin ID
                session['username'] = username
                session['full_name'] = 'Administrator'
                session['role'] = 'admin'
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.')
                return redirect(url_for('login'))
    
    return render_template('login/login.html')

@app.route('/logout')
def logout():
    # Log logout activity
    if db and session.get('user_id'):
        db.log_activity(session.get('user_id'), 'LOGOUT', f"User {session.get('username')} logged out")
    
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

@app.route('/dashboard')
@require_login
def dashboard():
    user_role = get_user_role()
    stats = {}
    
    if db and db.connection and db.connection.is_connected():
        user_id = session.get('user_id')
        # Get user's PDF history count (per-user)
        pdf_history = db.get_user_pdf_history(user_id)
        stats['my_pdfs'] = len(pdf_history)

        # Get recent activity
        recent_activity = db.get_activity_logs(user_id, 5)
        stats['recent_activity'] = recent_activity
    
    # (Optional) system-wide total PDFs if needed in future
    # total_pdf_history = db.get_all_pdf_history()  # not implemented
    
    return render_template('dashboard/dashboard.html', stats=stats, user_role=user_role)

@app.route('/settings', methods=['GET', 'POST'])
@require_login
def settings():
    """User settings page with enhanced functionality"""
    user_id = session.get('user_id')
    
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.', 'error')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        try:
            if action == 'change_password':
                old_password = request.form.get('old_password')
                new_password = request.form.get('new_password')
                confirm_password = request.form.get('confirm_password')
                
                if not old_password or not new_password:
                    flash('Please fill in all password fields.', 'error')
                elif new_password != confirm_password:
                    flash('New passwords do not match.', 'error')
                elif len(new_password) < 6:
                    flash('Password must be at least 6 characters long.', 'error')
                else:
                    success, message = db.change_password(user_id, old_password, new_password)
                    if success:
                        flash('Password changed successfully!', 'success')
                    else:
                        flash(message, 'error')
            
            elif action == 'update_profile':
                full_name = request.form.get('full_name', '').strip()
                email = request.form.get('email', '').strip()
                
                if not full_name or not email:
                    flash('Please fill in all profile fields.', 'error')
                elif '@' not in email:
                    flash('Please enter a valid email address.', 'error')
                else:
                    success, message = db.update_user_profile(user_id, full_name, email)
                    if success:
                        flash('Profile updated successfully!', 'success')
                        # Update session data
                        session['full_name'] = full_name
                        session['email'] = email
                    else:
                        flash(message, 'error')
            
            elif action == 'update_preferences':
                # Handle user preferences
                email_notifications = request.form.get('email_notifications') == 'on'
                auto_save = request.form.get('auto_save') == 'on'
                theme = request.form.get('theme', 'light')
                
                # Save preferences
                db.update_user_setting(user_id, 'email_notifications', str(email_notifications))
                db.update_user_setting(user_id, 'auto_save', str(auto_save))
                db.update_user_setting(user_id, 'theme', theme)
                
                flash('Preferences updated successfully!', 'success')
            
            else:
                flash('Invalid action specified.', 'error')
                
        except Exception as e:
            app.logger.error(f"Error in settings: {e}")
            flash(f'Error updating settings: {str(e)}', 'error')
    
    # Get current user data and settings
    try:
        user_data = db.get_user_by_id(user_id)
        
        # Check if database methods exist, if not use defaults
        if hasattr(db, 'get_user_settings'):
            user_settings = db.get_user_settings(user_id)
        else:
            # Default user settings if method doesn't exist
            user_settings = {
                'email_notifications': 'true',
                'auto_save': 'false',
                'theme': 'light'
            }
        
        # Create user_settings table if method exists
        if hasattr(db, 'create_user_settings_table'):
            db.create_user_settings_table()
        
        return render_template('settings/settings.html', 
                             user=user_data, 
                             settings=user_settings)
    except Exception as e:
        app.logger.error(f"Error loading settings page: {e}")
        flash(f'Error loading settings: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/admin')
@require_role('admin')
def admin_panel():
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('dashboard'))
    
    # Get all users
    all_users = db.get_users_by_role()
    
    # Get recent activity logs
    recent_logs = db.get_activity_logs(limit=20)
    
    return render_template('admin/admin.html', users=all_users, recent_logs=recent_logs)

@app.route('/admin/users')
@require_role('admin')
def manage_users():
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('dashboard'))
    
    all_users = db.get_users_by_role()
    return render_template('manage_users.html', users=all_users)

@app.route('/admin/update_role', methods=['POST'])
@require_role('admin')
def update_user_role():
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('admin_panel'))
    
    user_id = request.form.get('user_id')
    new_role = request.form.get('new_role')
    
    if user_id and new_role in ['admin', 'faculty', 'student']:
        updated_by = session.get('user_id')
        success = db.update_user_role(user_id, new_role, updated_by)
        
        if success:
            flash(f'User role updated to {new_role} successfully.')
        else:
            flash('Failed to update user role.')
    else:
        flash('Invalid user or role.')
    
    return redirect(url_for('manage_users'))

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if db and db.connection and db.connection.is_connected():
            success, result = db.create_password_reset_token(email)

            # Always show the generic message so we don't reveal whether the email exists
            flash('If the email exists, a password reset link has been sent.')

            # If token creation succeeded, send the reset email (result contains the token)
            if success and result:
                token = result
                try:
                    reset_link = url_for('reset_password_with_token', token=token, _external=True)
                    subject = 'Password reset request'
                    body = f"Hello,\n\nA password reset was requested for your account. If you made this request, click the link below to reset your password:\n\n{reset_link}\n\nIf you did not request this, you can safely ignore this email. The link will expire in 1 hour.\n\nRegards,\nExam Seating Plan Generator"

                    send_success, send_msg = db.send_email(email, subject, body)
                    if send_success:
                        app.logger.info(f"Password reset email sent to {email}")
                    else:
                        app.logger.error(f"Failed to send password reset email to {email}: {send_msg}")
                except Exception as e:
                    app.logger.error(f"Error while sending reset email: {e}")

            # Log the password reset request (do not include sensitive token in logs)
            db.log_activity(None, 'PASSWORD_RESET_REQUEST', f"Password reset requested for email: {email}")
        else:
            flash('Database not available.')
        
        return redirect(url_for('login'))
    
    return render_template('reset password/reset_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password_with_token(token):
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('reset_password_with_token', token=token))
        
        if db and db.connection and db.connection.is_connected():
            success, message = db.reset_password_with_token(token, new_password)
            flash(message)
            
            if success:
                return redirect(url_for('login'))
        else:
            flash('Database not available.')
        
        return redirect(url_for('reset_password_with_token', token=token))
    
    return render_template('reset password/reset_password_form.html', token=token)

@app.route('/pdf_history', methods=['GET', 'POST'])
@require_login
def pdf_history():
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('dashboard'))
    
    try:
        # Ensure tables are properly created/updated
        db.create_tables()
        
        # Get PDF history from database
        user_id = session.get('user_id')
        pdf_history_data = db.get_user_pdf_history(user_id)
        
        # Calculate stats
        total_students = sum(pdf.get('student_count', 0) for pdf in pdf_history_data)
        total_rooms = sum(pdf.get('room_count', 0) for pdf in pdf_history_data)
        
        # Log activity
        db.log_activity(user_id, 'PDF_HISTORY_VIEW', "User viewed PDF history")
        
        return render_template('pdf history/pdf_history.html',
                             pdf_history=pdf_history_data,
                             total_students=total_students,
                             total_rooms=total_rooms)
    
    except Exception as e:
        app.logger.error(f"Error fetching PDF history: {e}")
        print(f"Error fetching PDF history: {e}")  # Also print to console
        flash('Error loading PDF history. Please try again.')
        return redirect(url_for('dashboard'))

@app.route('/download_history_pdf/<int:pdf_id>')
@require_login
def download_history_pdf(pdf_id):
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('pdf_history'))
    
    try:
        user_id = session.get('user_id')
        file_path = db.get_pdf_file_path(pdf_id, user_id)
        
        if file_path and os.path.exists(file_path):
            # Log download activity
            db.log_activity(user_id, 'PDF_DOWNLOAD', f"Downloaded PDF ID: {pdf_id}")
            return send_file(file_path, as_attachment=True)
        else:
            flash('PDF file not found.')
            return redirect(url_for('pdf_history'))
    
    except Exception as e:
        app.logger.error(f"Error downloading PDF {pdf_id}: {e}")
        flash('Error downloading PDF file.')
        return redirect(url_for('pdf_history'))

@app.route('/', methods=['GET', 'POST'])
@require_login
def index():
    return redirect(url_for('dashboard'))

@app.route('/generate_plan', methods=['GET', 'POST'])
@require_login
def generate_plan():
    if request.method == 'GET':
        # Get default settings for the form
        default_settings = {
            'students_per_desk': 1,
            'include_detained': False,
            'default_building': 'Main Building'
        }
        
        return render_template('generate plan/generate_plan.html', 
                             default_settings=default_settings)
    
    try:
        # Handle file uploads
        student_csv_file = request.files.get('student_csv')
        room_csv_file = request.files.get('room_csv')
        
        if not student_csv_file or not room_csv_file:
            flash('Please upload both student and room CSV files.', 'error')
            return redirect(url_for('generate_plan'))
        
        # Get form data
        students_per_desk = int(request.form.get('students_per_desk', 1))
        include_detained = 'include_detained' in request.form
        fill_order = request.form.get('fill_order', 'row')
        building = request.form.get('building', 'Main Building')
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
        uploads_dir = os.path.abspath(uploads_dir)
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        student_filename = f"students_{timestamp}.csv"
        room_filename = f"rooms_{timestamp}.csv"
        
        student_path = os.path.join(uploads_dir, student_filename)
        room_path = os.path.join(uploads_dir, room_filename)
        
        student_csv_file.save(student_path)
        room_csv_file.save(room_path)
        
        # Store paths in session for PDF generation
        session['last_student_file'] = student_path
        session['last_room_file'] = room_path
        session['last_students_per_desk'] = students_per_desk
        session['last_include_detained'] = include_detained
        session['last_building'] = building
        
        print(f"Generating seating plan with {students_per_desk} students per desk, include_detained: {include_detained}")
        
        # Import here to avoid circular imports
        from backend.utils.seating import generate_seating_plan
        
        # Generate seating plan
        result, unseated_students, error = generate_seating_plan(
            student_path, room_path, students_per_desk, include_detained, True, fill_order
        )
        
        if error:
            flash(f'Error generating seating plan: {error}', 'error')
            return render_template('generate plan/generate_plan.html', 
                                 default_settings={'students_per_desk': students_per_desk, 
                                                 'include_detained': include_detained, 
                                                 'default_building': building})
        
        if result is None or not result:
            flash('Failed to generate seating plan', 'error')
            return render_template('generate plan/generate_plan.html', 
                                 default_settings={'students_per_desk': students_per_desk, 
                                                 'include_detained': include_detained, 
                                                 'default_building': building})
        
        # Unpack result - result is now a tuple (seating_plan, summary_stats)
        if isinstance(result, tuple) and len(result) == 2:
            seating_plan, summary_stats = result
        else:
            # Fallback for unexpected format
            seating_plan = result if isinstance(result, list) else []
            # Calculate summary stats manually as fallback
            total_seated = 0
            for room in seating_plan:
                if isinstance(room, dict) and 'students_count' in room:
                    total_seated += room['students_count']
            
            summary_stats = {
                'total_students_available': 0,
                'total_students_seated': total_seated,
                'total_unseated': len(unseated_students) if unseated_students else 0,
                'total_capacity': sum(room.get('capacity', 0) for room in seating_plan if isinstance(room, dict)),
                'total_rooms': len([r for r in seating_plan if isinstance(r, dict) and r.get('students_count', 0) > 0]),
                'overall_utilization': 0,
                'branch_distribution': {},
                'students_per_desk': students_per_desk,
                'include_detained': include_detained
            }
        
        # Store in session for PDF generation
        session['seating_plan'] = seating_plan
        session['summary_stats'] = summary_stats
        session['unseated_students'] = unseated_students
        session['generation_settings'] = {
            'students_per_desk': students_per_desk,
            'include_detained': include_detained
        }

        # Also remember fill_order for later PDF generation
        session['generation_settings']['fill_order'] = fill_order
        
        # Log the generation
        user_id = session.get('user_id')
        if db and user_id:
            db.log_activity(
                user_id, 
                'SEATING_PLAN_GENERATED', 
                f'Generated seating plan with {summary_stats["total_students_seated"]} students in {summary_stats["total_rooms"]} rooms',
                get_client_ip(),
                get_user_agent()
            )
        
        flash(f'Seating plan generated successfully! {summary_stats["total_students_seated"]} students seated in {summary_stats["total_rooms"]} rooms.', 'success')
        
        return render_template('seating plan/seating_plan.html', 
                             seating_plan=seating_plan,
                             summary_stats=summary_stats,
                             unseated_students=unseated_students)
        
    except Exception as e:
        print(f"Error in generate_plan: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error generating seating plan: {str(e)}', 'error')
        return render_template('generate plan/generate_plan.html', 
                             default_settings={'students_per_desk': 1, 
                                             'include_detained': False, 
                                             'default_building': 'Main Building'})

@app.route('/download_pdf', methods=['GET', 'POST'])
@require_login
def download_pdf():
    # Get the last uploaded files from session
    student_csv = session.get('last_student_file')
    room_csv = session.get('last_room_file')
    students_per_desk = session.get('last_students_per_desk', 1)
    include_detained = session.get('last_include_detained', False)
    building = session.get('last_building', 'Main Building')
    
    if not student_csv or not room_csv:
        flash('No files found for PDF export. Please upload CSV files first.')
        return redirect(url_for('generate_plan'))
    
    if not os.path.exists(student_csv) or not os.path.exists(room_csv):
        flash('CSV files not found for PDF export. Please upload files again.')
        return redirect(url_for('generate_plan'))
    
    from backend.utils.seating import generate_seating_plan
    
    # Debug: Print parameters being passed
    print(f"Generating seating plan with:")
    print(f"  Student CSV: {student_csv}")
    print(f"  Room CSV: {room_csv}")
    print(f"  Students per desk: {students_per_desk}")
    print(f"  Include detained: {include_detained}")
    
    # Use previously selected fill_order if available
    generation_settings = session.get('generation_settings', {})
    fill_order = generation_settings.get('fill_order', 'row')

    generated_result, unseated_students, error = generate_seating_plan(
        student_csv, room_csv, students_per_desk, include_detained, True, fill_order
    )

    if error:
        flash(f'Error generating seating plan: {error}')
        return redirect(url_for('generate_plan'))

    if isinstance(generated_result, tuple) and len(generated_result) == 2:
        seating_plan, summary_stats = generated_result
    else:
        seating_plan = generated_result if isinstance(generated_result, list) else []
        summary_stats = session.get('summary_stats', {})

    print(f"Generated seating plan with {len(seating_plan)} rooms:")
    for idx, room in enumerate(seating_plan):
        if isinstance(room, dict):
            print(f"  Room {idx + 1}: {room.get('room_number')} / {room.get('room_name')} - {room.get('students_count', 0)} students")

    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        pdf_folder = os.path.join(base_dir, '..', 'data', 'pdfs')
        pdf_folder = os.path.abspath(pdf_folder)
        os.makedirs(pdf_folder, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"seating_plan_{timestamp}.pdf"

        pdf_path = export_to_pdf(seating_plan, pdf_filename, pdf_folder)

        # Save to database if available
        if db and db.connection and db.connection.is_connected():
            user_id = session.get('user_id')

            # Count students and rooms
            student_count = sum(room.get('students_count', 0) for room in seating_plan if isinstance(room, dict))
            room_count = len([room for room in seating_plan if isinstance(room, dict)])

            db.save_pdf_history(
                user_id, pdf_filename, pdf_path, 
                student_count, room_count, 
                students_per_desk, include_detained, building
            )
            
            # Log PDF generation
            db.log_activity(user_id, 'PDF_GENERATED', f"Generated seating plan PDF: {pdf_filename}")

        flash(f'PDF generated successfully with {len(seating_plan)} rooms!')
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
        
    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}")
        flash(f'Error generating PDF: {str(e)}')
        return redirect(url_for('generate_plan'))

@app.route('/send_email_notifications', methods=['POST'])
@require_login
def send_email_notifications():
    if not db or not db.connection or not db.connection.is_connected():
        flash('Database not available.')
        return redirect(url_for('pdf_history'))
    
    try:
        pdf_id = request.form.get('pdf_id')
        recipient_type = request.form.get('recipient_type')
        
        if not pdf_id or not recipient_type:
            flash('Invalid request parameters.')
            return redirect(url_for('pdf_history'))
        
        # Get PDF details
        user_id = session.get('user_id')
        pdf_path = db.get_pdf_file_path(pdf_id, user_id)
        
        if not pdf_path or not os.path.exists(pdf_path):
            flash('PDF file not found.')
            return redirect(url_for('pdf_history'))
        
        # Get email addresses based on recipient type
        recipients = []
        
        # First try to get emails from CSV files (session or sample)
        csv_emails = []
        
        # Try session stored file first
        last_student_file = session.get('last_student_file')
        if last_student_file:
            csv_emails = db.get_student_emails_from_session_file(last_student_file)
            app.logger.info(f"Found {len(csv_emails)} emails from session file: {last_student_file}")
        
        # If no session file or no emails found, try sample file
        if not csv_emails:
            sample_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'sample_students.csv')
            sample_file_path = os.path.abspath(sample_file_path)
            
            if os.path.exists(sample_file_path):
                csv_emails = db.get_emails_from_csv_data(sample_file_path)
                app.logger.info(f"Found {len(csv_emails)} emails from sample file: {sample_file_path}")
            else:
                app.logger.warning(f"Sample file not found: {sample_file_path}")
        
        if recipient_type == 'students':
            # Try database first, then CSV
            db_emails = db.get_all_emails_by_role('student')
            if db_emails:
                recipients = db_emails
                app.logger.info(f"Using {len(db_emails)} database student emails")
            else:
                recipients = csv_emails
                app.logger.info(f"Using {len(csv_emails)} CSV student emails")
                
        elif recipient_type == 'faculty':
            # Only use database for faculty
            recipients = db.get_all_emails_by_role('faculty')
            app.logger.info(f"Using {len(recipients)} database faculty emails")
            
        elif recipient_type == 'both':
            # Get students from CSV or database
            student_emails = db.get_all_emails_by_role('student')
            if not student_emails:
                student_emails = csv_emails
                
            # Get faculty from database only
            faculty_emails = db.get_all_emails_by_role('faculty')
            recipients = student_emails + faculty_emails
            
            app.logger.info(f"Using {len(student_emails)} student + {len(faculty_emails)} faculty emails")
        
        # Remove duplicates and filter empty emails
        recipients = list(set([email for email in recipients if email and email.strip() and '@' in email]))
        
        # Debug logging
        app.logger.info(f"Recipient type: {recipient_type}")
        app.logger.info(f"Final recipients ({len(recipients)}): {recipients}")
        
        if not recipients:
            error_msg = f'No email addresses found for {recipient_type}.'
            if recipient_type == 'students':
                error_msg += ' Ensure your CSV file has an "email" column with valid email addresses.'
            else:
                error_msg += ' Ensure database has users with the appropriate role.'
            
            flash(error_msg)
            return redirect(url_for('pdf_history'))
        
        # Send emails (simulated)
        success_count = 0
        error_count = 0
        
        for email in recipients:
            subject = "Exam Seating Plan Notification"
            body = f"""
Dear Student/Faculty,

Please find the exam seating plan generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

Please review the seating arrangement and contact the administration if you have any questions.

Best regards,
Exam Management System
"""
            
            success, error_msg = db.send_email(email, subject, body, pdf_path=pdf_path)
            
            if success:
                success_count += 1
                db.log_activity(user_id, 'EMAIL_SENT', f"Seating plan sent to {email}")
            else:
                error_count += 1
                app.logger.error(f"Failed to send email to {email}: {error_msg}")
        
        # Provide feedback to user
        if success_count > 0:
            flash(f'Successfully sent {success_count} email notification(s) to: {", ".join(recipients[:3])}{"..." if len(recipients) > 3 else ""}')
        
        if success_count == 0:
            flash('No emails could be sent. Please check the logs.')
            
    except Exception as e:
        app.logger.error(f"Error in send_email_notifications: {e}")
        flash(f'Error sending email notifications: {str(e)}')
    
    return redirect(url_for('pdf_history'))

# Add debug route to check CSV emails
@app.route('/debug/csv_emails')
@require_login
def debug_csv_emails():
    """Debug route to check CSV email extraction"""
    try:
        results = {}
        
        # Check session file
        last_student_file = session.get('last_student_file')
        if last_student_file:
            session_emails = db.get_student_emails_from_session_file(last_student_file)
            results['session_file'] = last_student_file
            results['session_emails'] = session_emails
        else:
            results['session_file'] = 'None'
            results['session_emails'] = []
        
        # Check sample file
        sample_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'sample_students.csv')
        sample_file_path = os.path.abspath(sample_file_path)
        
        if os.path.exists(sample_file_path):
            sample_emails = db.get_emails_from_csv_data(sample_file_path)
            results['sample_file'] = sample_file_path
            results['sample_emails'] = sample_emails
        else:
            results['sample_file'] = f'Not found: {sample_file_path}'
            results['sample_emails'] = []
        
        # Check database emails
        db_student_emails = db.get_all_emails_by_role('student')
        db_faculty_emails = db.get_all_emails_by_role('faculty')
        
        results['db_student_emails'] = db_student_emails
        results['db_faculty_emails'] = db_faculty_emails
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ...existing code...

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Log application startup information
    print("=" * 60)
    print("🚀 EXAM SEATING PLAN GENERATOR - STARTING UP")
    print("=" * 60)
    print(f"📍 Application URL: http://localhost:5000")
    print(f"📍 Local Network URL: http://0.0.0.0:5000")
    print(f"🔧 Debug Mode: Enabled")
    print(f"📁 Project Directory: {os.getcwd()}")
    print("=" * 60)
    print("🌐 Available Routes:")
    print("   • Login: http://localhost:5000/login")
    print("   • Signup: http://localhost:5000/signup")
    print("   • Dashboard: http://localhost:5000/dashboard")
    print("   • Admin Panel: http://localhost:5000/admin")
    print("=" * 60)
    
    # Change host to your desired IP, e.g., '0.0.0.0' for all interfaces or your LAN IP
    app.run(host='0.0.0.0', port=5000, debug=True)

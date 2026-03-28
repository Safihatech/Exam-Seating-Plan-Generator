import mysql.connector
from mysql.connector import Error
import bcrypt
import smtplib
import csv
import os
import secrets
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import sqlite3
import bcrypt
from datetime import datetime
import os

from email.mime.base import MIMEBase
from email import encoders
import time
class Database:
    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.password = ''
        self.database = 'seating_plan_db'
        
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_username = "seatgenerator.gprec@gmail.com"
        self.email_password = "jhpj lkkv rjmp fzfc"
        
        self.connection = None
        self.connect()

    def connect(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
                return True
            
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            self.connection = None
            return False

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")

    def create_user(self, username, email, password, full_name, role='student'):
        """Create a new user"""
        try:
            cursor = self.connection.cursor()
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            query = """
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (username, email, password_hash, full_name, role))
            self.connection.commit()
            
            user_id = cursor.lastrowid
            print(f"User created successfully with ID: {user_id}")
            return True, user_id
            
        except Error as e:
            print(f"Error creating user: {e}")
            self.connection.rollback()
            return False, str(e)
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def authenticate_user(self, identifier, password):
        """Authenticate user login by username or email"""
        try:
            cursor = self.connection.cursor()

            query = """
            SELECT id, username, email, password_hash, full_name, role, is_active
            FROM users
            WHERE (username = %s OR email = %s) AND is_active = TRUE
            """
            # Use the provided identifier for both fields (username or email)
            cursor.execute(query, (identifier, identifier))
            user = cursor.fetchone()

            if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
                # Log successful login
                display_name = user[1] or user[2] or identifier
                self.log_activity(user[0], 'LOGIN_SUCCESS', f'User {display_name} logged in successfully')
                return user
            else:
                # Log failed login attempt; do not expose whether username or email exists
                self.log_activity(None, 'LOGIN_FAILED', f'Failed login attempt for identifier: {identifier}')
                return None

        except Error as e:
            print(f"Error authenticating user: {e}")
            return None
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_user_by_id(self, user_id):
        """Get user information by ID"""
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT id, username, email, full_name, role, is_active, created_at
            FROM users 
            WHERE id = %s
            """
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
            
        except Error as e:
            print(f"Error getting user by ID: {e}")
            return None
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def create_tables(self):
        """Create all necessary tables"""
        try:
            cursor = self.connection.cursor()
            
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                role ENUM('admin', 'faculty', 'student') DEFAULT 'student',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """)
            
            # Students table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                program VARCHAR(50),
                year INT,
                semester INT,
                detained BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # PDF History table - updated with pdf_path column
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                pdf_path VARCHAR(500),
                student_count INT DEFAULT 0,
                room_count INT DEFAULT 0,
                students_per_desk INT DEFAULT 1,
                include_detained BOOLEAN DEFAULT FALSE,
                building VARCHAR(100) DEFAULT 'Main Building',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)
            
            # Activity logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                action VARCHAR(100) NOT NULL,
                description TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """)
            
            # Settings table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                setting_key VARCHAR(100) UNIQUE NOT NULL,
                setting_value TEXT,
                updated_by INT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
            )
            """)
            
            # Password reset tokens table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                token VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)
            
            # Check if pdf_path column exists, if not add it
            cursor.execute("""
            SELECT COUNT(*) as count
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'pdf_history' 
            AND COLUMN_NAME = 'pdf_path'
            """)
            
            result = cursor.fetchone()
            if result[0] == 0:
                cursor.execute("""
                ALTER TABLE pdf_history 
                ADD COLUMN pdf_path VARCHAR(500) AFTER filename
                """)
                print("Added pdf_path column to pdf_history table")
            
            self.connection.commit()
            cursor.close()
            print("Database tables created/updated successfully.")
            return True
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False

    def save_pdf_history(self, user_id, filename, pdf_path, student_count, room_count, students_per_desk, include_detained, building):
        """Save PDF generation history"""
        try:
            cursor = self.connection.cursor()
            query = """
            INSERT INTO pdf_history (user_id, filename, pdf_path, student_count, room_count, 
                                   students_per_desk, include_detained, building)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (user_id, filename, pdf_path, student_count, room_count, 
                                 students_per_desk, include_detained, building))
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving PDF history: {e}")
            return False

    def get_user_pdf_history(self, user_id):
        """Get PDF history for a specific user"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT id, filename, pdf_path, student_count, room_count, 
                   students_per_desk, building, created_at
            FROM pdf_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC
            """
            cursor.execute(query, (user_id,))
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error fetching PDF history: {e}")
            return []

    def get_pdf_file_path(self, pdf_id, user_id):
        """Get PDF file path by ID and user ID"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT pdf_path FROM pdf_history WHERE id = %s AND user_id = %s"
            cursor.execute(query, (pdf_id, user_id))
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching PDF file path: {e}")
            return None

    def get_users_by_role(self, role=None):
        """Get all users or users filtered by role"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            if role:
                query = "SELECT id, username, email, full_name, role, is_active, created_at FROM users WHERE role = %s AND is_active = TRUE ORDER BY created_at DESC"
                cursor.execute(query, (role,))
            else:
                query = "SELECT id, username, email, full_name, role, is_active, created_at FROM users WHERE is_active = TRUE ORDER BY created_at DESC"
                cursor.execute(query)
        
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error fetching users: {e}")
            return []

    def get_all_student_emails(self):
        """Get all student email addresses"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT email FROM users WHERE role = 'student' AND is_active = TRUE"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return [row[0] for row in results if row[0]]  # Filter out None emails
        except Exception as e:
            print(f"Error fetching student emails: {e}")
            return []

    def get_all_faculty_emails(self):
        """Get all faculty email addresses"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT email FROM users WHERE role = 'faculty' AND is_active = TRUE"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return [row[0] for row in results if row[0]]  # Filter out None emails
        except Exception as e:
            print(f"Error fetching faculty emails: {e}")
            return []

    def get_all_emails_by_role(self, role):
        """Get all email addresses for a specific role"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT email FROM users WHERE role = %s AND is_active = TRUE AND email IS NOT NULL AND email != ''"
            cursor.execute(query, (role,))
            results = cursor.fetchall()
            cursor.close()
            
            emails = [row[0] for row in results if row[0] and '@' in row[0]]
            print(f"Found {len(emails)} {role} emails from database: {emails}")
            return emails
            
        except Exception as e:
            print(f"Error fetching {role} emails: {e}")
            return []

    def send_email(self, to_email, subject, body, pdf_path=None):
        """Send email notification with proper SMTP integration and logging"""
        try:
            # Validate email configuration
            if not all([self.smtp_server, self.smtp_port, self.email_username, self.email_password]):
                error_msg = "Email configuration incomplete. Please check SMTP settings."
                self.log_activity(None, 'EMAIL_ERROR', f'Failed to send email to {to_email}: {error_msg}')
                return False, error_msg

            # Validate recipient email
            if not to_email or '@' not in to_email:
                error_msg = f"Invalid recipient email address: {to_email}"
                self.log_activity(None, 'EMAIL_ERROR', error_msg)
                return False, error_msg

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add body to email
            msg.attach(MIMEText(body, 'plain'))

            # Add PDF attachment if provided
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as attachment:
                        
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(pdf_path)}'
                        )
                        msg.attach(part)
                        
                except Exception as e:
                    error_msg = f"Failed to attach PDF file: {str(e)}"
                    self.log_activity(None, 'EMAIL_ERROR', f'PDF attachment error for {to_email}: {error_msg}')
                    print(f"Warning: {error_msg}")

            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Enable encryption
                server.login(self.email_username, self.email_password)
                
                # Send email
                text = msg.as_string()
                server.sendmail(self.email_username, to_email, text)
                
                # Log successful email
                attachment_info = f" with PDF attachment ({os.path.basename(pdf_path)})" if pdf_path else ""
                success_msg = f"Email sent successfully to {to_email}{attachment_info}"
                self.log_activity(None, 'EMAIL_SENT', success_msg)
                print(success_msg)
                
                return True, success_msg

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP Authentication failed. Please check email credentials: {str(e)}"
            self.log_activity(None, 'EMAIL_ERROR', f'SMTP Auth error for {to_email}: {error_msg}')
            print(f"Email error: {error_msg}")
            return False, error_msg

        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"Recipient email refused: {to_email}"
            self.log_activity(None, 'EMAIL_ERROR', f'Recipient refused: {error_msg}')
            print(f"Email error: {error_msg}")
            return False, error_msg

        except smtplib.SMTPServerDisconnected as e:
            error_msg = f"SMTP server disconnected: {str(e)}"
            self.log_activity(None, 'EMAIL_ERROR', f'SMTP disconnected for {to_email}: {error_msg}')
            print(f"Email error: {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error sending email to {to_email}: {str(e)}"
            self.log_activity(None, 'EMAIL_ERROR', error_msg)
            print(f"Email error: {error_msg}")
            return False, error_msg

    def send_bulk_email(self, recipient_list, subject, body, pdf_path=None):
        """Send email to multiple recipients with detailed logging"""
        try:
            if not recipient_list:
                error_msg = "No recipients provided for bulk email"
                self.log_activity(None, 'EMAIL_ERROR', error_msg)
                return False, error_msg, []

            successful_emails = []
            failed_emails = []
            
            total_recipients = len(recipient_list)
            self.log_activity(None, 'EMAIL_BULK_START', f'Starting bulk email to {total_recipients} recipients')
            
            for i, email in enumerate(recipient_list, 1):
                try:
                    success, message = self.send_email(email, subject, body, pdf_path)
                    
                    if success:
                        successful_emails.append(email)
                        print(f"[{i}/{total_recipients}] ✓ Sent to {email}")
                    else:
                        failed_emails.append({'email': email, 'error': message})
                        print(f"[{i}/{total_recipients}] ✗ Failed to {email}: {message}")
                        
                    # Small delay to avoid overwhelming the SMTP server
                    time.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"Exception sending to {email}: {str(e)}"
                    failed_emails.append({'email': email, 'error': error_msg})
                    print(f"[{i}/{total_recipients}] ✗ Exception for {email}: {error_msg}")

            # Log bulk email completion
            success_count = len(successful_emails)
            failure_count = len(failed_emails)
            
            completion_msg = f"Bulk email completed: {success_count} successful, {failure_count} failed out of {total_recipients} total"
            self.log_activity(None, 'EMAIL_BULK_COMPLETE', completion_msg)
            
            if failure_count > 0:
                failed_list = [f"{item['email']} ({item['error']})" for item in failed_emails[:5]]  # Log first 5 failures
                failure_details = f"Failed emails: {'; '.join(failed_list)}"
                if failure_count > 5:
                    failure_details += f" and {failure_count - 5} more..."
                self.log_activity(None, 'EMAIL_BULK_FAILURES', failure_details)

            return True, completion_msg, {
                'successful': successful_emails,
                'failed': failed_emails,
                'total': total_recipients,
                'success_count': success_count,
                'failure_count': failure_count
            }

        except Exception as e:
            error_msg = f"Bulk email process failed: {str(e)}"
            self.log_activity(None, 'EMAIL_ERROR', error_msg)
            return False, error_msg, []

    def log_activity(self, user_id, action, details, ip_address=None, user_agent=None):
        """Log user activity"""
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (user_id, action, details, ip_address, user_agent))
            self.connection.commit()
            
        except Error as e:
            print(f"Error logging activity: {e}")
            self.connection.rollback()
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_students(self, include_detained=False):
        """Get all students"""
        try:
            cursor = self.connection.cursor()
            
            if include_detained:
                query = "SELECT roll_number, name, branch, section, is_detained, email FROM students ORDER BY roll_number"
            else:
                query = "SELECT roll_number, name, branch, section, is_detained, email FROM students WHERE is_detained = FALSE ORDER BY roll_number"
            
            cursor.execute(query)
            return cursor.fetchall()
            
        except Error as e:
            print(f"Error getting students: {e}")
            return []
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_setting(self, key, default_value=None):
        """Get system setting"""
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT setting_value FROM system_settings WHERE setting_key = %s"
            cursor.execute(query, (key,))
            result = cursor.fetchone()
            
            return result[0] if result else default_value
            
        except Error as e:
            print(f"Error getting setting {key}: {e}")
            return default_value
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def set_setting(self, key, value, description=None):
        """Set system setting"""
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO system_settings (setting_key, setting_value, description)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            setting_value = VALUES(setting_value),
            description = VALUES(description)
            """
            cursor.execute(query, (key, value, description))
            self.connection.commit()
            
            return True
            
        except Error as e:
            print(f"Error setting {key}: {e}")
            self.connection.rollback()
            return False
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_dashboard_stats(self, user_id=None):
        """Get dashboard statistics"""
        try:
            cursor = self.connection.cursor()
            stats = {}
            
            print(f"Getting dashboard stats for user_id: {user_id}")
            
            # Total students
            cursor.execute("SELECT COUNT(*) FROM students")
            result = cursor.fetchone()
            stats['total_students'] = result[0] if result else 0
            
            # Students by branch
            cursor.execute("SELECT branch, COUNT(*) FROM students GROUP BY branch")
            branch_data = cursor.fetchall()
            stats['students_by_branch'] = dict(branch_data) if branch_data else {}
            
            # Total PDFs generated
            if user_id:
                cursor.execute("SELECT COUNT(*) FROM pdf_history WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM pdf_history")
            result = cursor.fetchone()
            stats['total_pdfs'] = result[0] if result else 0
            
            # Recent activity count
            if user_id:
                cursor.execute("SELECT COUNT(*) FROM activity_logs WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM activity_logs")
            result = cursor.fetchone()
            stats['total_activities'] = result[0] if result else 0
            
            # Total users (admin only)
            if not user_id:
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
                result = cursor.fetchone()
                stats['total_users'] = result[0] if result else 0
            
            return stats
            
        except Error as e:
            print(f"Error getting dashboard stats: {e}")
            return {
                'total_students': 0,
                'total_pdfs': 0,
                'total_activities': 0,
                'total_users': 0,
                'students_by_branch': {}
            }
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    
    def change_password(self, user_id, old_password, new_password):
        """Change user password after verifying old password"""
        try:
            # Get current password hash
            cursor = self.connection.cursor()
            cursor.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            
            if not result:
                return False, "User not found"
            
            current_password_hash = result[0]
            
            # Ensure we have a valid hash
            if not current_password_hash:
                return False, "Invalid password hash in database"
            
            # Validate bcrypt hash format
            if not isinstance(current_password_hash, str) or len(current_password_hash) < 60:
                return False, "Invalid password hash format in database"
            
            # Check if hash starts with bcrypt identifier
            if not current_password_hash.startswith('$2'):
                return False, "Password hash is not in bcrypt format"
            
            # Convert hash to bytes if it's a string
            if isinstance(current_password_hash, str):
                hash_bytes = current_password_hash.encode('utf-8')
            else:
                hash_bytes = current_password_hash
            
            # Verify old password with proper error handling
            try:
                if not bcrypt.checkpw(old_password.encode('utf-8'), hash_bytes):
                    return False, "Current password is incorrect"
            except ValueError as ve:
                print(f"Password verification error: {ve}")
                return False, "Password hash is corrupted or invalid format"
            except Exception as e:
                print(f"Unexpected bcrypt error: {e}")
                return False, "Password verification failed"
            
            # Hash new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Update password
            cursor.execute(
                "UPDATE users SET password_hash = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (new_password_hash, user_id)
            )
            self.connection.commit()
            
            return True, "Password changed successfully"
            
        except Exception as e:
            print(f"Error changing password: {e}")
            return False, "Failed to change password"
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def create_password_reset_token(self, email, expires_hours=1):
        """Create a password reset token for a user identified by email.

        Returns (True, token) on success or (False, message) on failure.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT id FROM users WHERE email = %s AND is_active = TRUE", (email,))
            row = cursor.fetchone()
            if not row:
                # Do not reveal whether email exists in production - keep message generic
                return False, "If the email exists, a password reset link will be sent."

            user_id = row[0]

            token = secrets.token_urlsafe(48)
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

            insert_query = """
            INSERT INTO password_reset_tokens (user_id, token, expires_at, used)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, token, expires_at.strftime('%Y-%m-%d %H:%M:%S'), False))
            self.connection.commit()
            cursor.close()

            # Log activity without leaking sensitive info
            try:
                self.log_activity(user_id, 'PASSWORD_RESET_CREATED', f'Password reset token created for user_id {user_id}')
            except Exception:
                pass

            return True, token

        except Exception as e:
            print(f"Error creating password reset token: {e}")
            try:
                if self.connection:
                    self.connection.rollback()
            except Exception:
                pass
            return False, str(e)

    def reset_password_with_token(self, token, new_password):
        """Reset a user's password using a password reset token.

        Returns (True, message) on success, (False, message) on failure.
        """
        try:
            cursor = self.connection.cursor()
            query = "SELECT id, user_id, expires_at, used FROM password_reset_tokens WHERE token = %s"
            cursor.execute(query, (token,))
            row = cursor.fetchone()

            if not row:
                return False, "Invalid or expired password reset token."

            token_id, user_id, expires_at, used = row[0], row[1], row[2], row[3]

            # Check if already used
            if used:
                return False, "This password reset link has already been used."

            # expires_at may be a datetime or string depending on connector; normalize
            if isinstance(expires_at, str):
                try:
                    expires_at_dt = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    expires_at_dt = datetime.utcnow()  # Force expiry if unparsable
            else:
                expires_at_dt = expires_at

            if expires_at_dt < datetime.utcnow():
                return False, "Password reset token has expired."

            # Hash the new password and update user
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            update_user_q = "UPDATE users SET password_hash = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            cursor.execute(update_user_q, (new_password_hash, user_id))

            # Mark token as used
            update_token_q = "UPDATE password_reset_tokens SET used = TRUE WHERE id = %s"
            cursor.execute(update_token_q, (token_id,))

            self.connection.commit()
            cursor.close()

            try:
                self.log_activity(user_id, 'PASSWORD_RESET_COMPLETED', f'Password reset completed for user_id {user_id}')
            except Exception:
                pass

            return True, "Password has been reset successfully."

        except Exception as e:
            print(f"Error resetting password with token: {e}")
            try:
                if self.connection:
                    self.connection.rollback()
            except Exception:
                pass
            return False, str(e)


    def get_system_stats(self):
        """Get comprehensive system statistics for admin dashboard"""
        try:
            cursor = self.connection.cursor()
            stats = {}
            
            # User statistics
            cursor.execute("SELECT role, COUNT(*) FROM users WHERE is_active = TRUE GROUP BY role")
            user_roles_data = cursor.fetchall()
            stats['user_roles'] = dict(user_roles_data) if user_roles_data else {}
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
            result = cursor.fetchone()
            stats['total_users'] = result[0] if result else 0
            
            # Student statistics
            cursor.execute("SELECT COUNT(*) FROM students")
            result = cursor.fetchone()
            stats['total_students'] = result[0] if result else 0
            
            cursor.execute("SELECT COUNT(*) FROM students WHERE is_detained = TRUE")
            result = cursor.fetchone()
            stats['detained_students'] = result[0] if result else 0
            
            cursor.execute("SELECT branch, COUNT(*) FROM students GROUP BY branch")
            branch_data = cursor.fetchall()
            stats['students_by_branch'] = dict(branch_data) if branch_data else {}
            
            # PDF generation statistics
            cursor.execute("SELECT COUNT(*) FROM pdf_history")
            result = cursor.fetchone()
            stats['total_pdfs'] = result[0] if result else 0
            
            # Activity statistics
            cursor.execute("SELECT COUNT(*) FROM activity_logs")
            result = cursor.fetchone()
            stats['total_activities'] = result[0] if result else 0
            
            return stats
            
        except Error as e:
            print(f"Error getting system stats: {e}")
            return {
                'total_users': 0,
                'total_students': 0,
                'total_pdfs': 0,
                'total_activities': 0,
                'detained_students': 0,
                'user_roles': {},
                'students_by_branch': {}
            }
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_recent_activities(self, limit=50):
        """Get recent activities for dashboard"""
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT al.id, u.username, u.full_name, al.action, al.details, 
                   al.ip_address, al.created_at
            FROM activity_logs al
            LEFT JOIN users u ON al.user_id = u.id
            ORDER BY al.created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            
            activities = []
            for row in cursor.fetchall():
                activities.append({
                    'id': row[0],
                    'username': row[1] or 'System',
                    'full_name': row[2] or 'System',
                    'action': row[3] or 'Unknown Action',
                    'details': row[4] or 'No details',
                    'ip_address': row[5] or 'Unknown IP',
                    'created_at': row[6]
                })
            
            return activities
            
        except Error as e:
            print(f"Error getting recent activities: {e}")
            return []
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def get_activity_logs(self, user_id, limit=50):
        """Get activity logs for specific user"""
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT id, action, details, ip_address, created_at
            FROM activity_logs 
            WHERE user_id = %s
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            
            activities = []
            for row in cursor.fetchall():
                activities.append({
                    'id': row[0],
                    'action': row[1] or 'Unknown Action',
                    'details': row[2] or 'No details',
                    'ip_address': row[3] or 'Unknown IP',
                    'created_at': row[4]
                })
            
            return activities
            
        except Error as e:
            print(f"Error getting activity logs: {e}")
            return []
        finally:
            try:
                if cursor:
                    cursor.close()
            except NameError:
                pass

    def import_students_from_csv(self, csv_file_path):
        """Import students from CSV file into database"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Find detained column
            detained_column = None
            for col in df.columns:
                if 'detained' in col.lower():
                    detained_column = col
                    break
        
            cursor = self.connection.cursor()
            inserted_count = 0
            
            for _, row in df.iterrows():
                try:
                    student_id = str(row.get('roll_number', row.get('id', f'STU_{inserted_count}')))
                    name = str(row.get('name', 'Unknown'))
                    program = str(row.get('branch', row.get('department', 'N/A')))
                    year = int(row.get('year', 1)) if pd.notna(row.get('year')) else 1
                    semester = int(row.get('semester', 1)) if pd.notna(row.get('semester')) else 1
                    
                    # Handle detained status properly
                    detained = False
                    if detained_column and pd.notna(row.get(detained_column)):
                        detained_value = str(row.get(detained_column)).upper()
                        detained = detained_value in ['TRUE', '1', 'YES', 'Y']
                
                    # Check if student already exists
                    check_query = "SELECT id FROM students WHERE student_id = %s"
                    cursor.execute(check_query, (student_id,))
                
                    if cursor.fetchone():
                        # Update existing student
                        update_query = """
                        UPDATE students 
                        SET name = %s, program = %s, year = %s, semester = %s, detained = %s
                        WHERE student_id = %s
                        """
                        cursor.execute(update_query, (name, program, year, semester, detained, student_id))
                    else:
                        # Insert new student
                        insert_query = """
                        INSERT INTO students (student_id, name, program, year, semester, detained)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (student_id, name, program, year, semester, detained))
                
                    inserted_count += 1
                
                except Exception as row_error:
                    print(f"Error processing row {inserted_count}: {row_error}")
                    continue
        
            self.connection.commit()
            cursor.close()
        
            print(f"Successfully processed {inserted_count} students from CSV")
            return inserted_count
        
        except Exception as e:
            print(f"Error importing students from CSV: {e}")
            if hasattr(self, 'connection') and self.connection:
                self.connection.rollback()
            return 0

    def get_duplicate_students(self, csv_file_path):
        """Check for duplicate roll numbers in CSV vs database"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("SELECT roll_number FROM students")
            existing_rolls = set(row[0] for row in cursor.fetchall())
            
            duplicates = []
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_num, row in enumerate(reader, start=2):
                    roll_number = row.get('roll_number', '').strip()
                    if roll_number and roll_number in existing_rolls:
                        duplicates.append({
                            'row': row_num,
                            'roll_number': roll_number,
                            'name': row.get('name', '').strip(),
                            'branch': row.get('branch', '').strip(),
                            'section': row.get('section', '').strip()
                        })
            
            return duplicates
            
        except Exception as e:
            print(f"Error checking duplicates: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def clear_all_students(self):
        """Clear all students from database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM students")
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            print(f"Cleared {deleted_count} students from database")
            return True, f"Cleared {deleted_count} students successfully"
            
        except Error as e:
            print(f"Error clearing students: {e}")
            self.connection.rollback()
            return False, f"Database error: {str(e)}"
        finally:
            if cursor:
                cursor.close()

    def get_users_by_role(self, role=None):
        """Get all users or users filtered by role"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            if role:
                query = "SELECT id, username, email, full_name, role, is_active, created_at FROM users WHERE role = %s AND is_active = TRUE ORDER BY created_at DESC"
                cursor.execute(query, (role,))
            else:
                query = "SELECT id, username, email, full_name, role, is_active, created_at FROM users WHERE is_active = TRUE ORDER BY created_at DESC"
                cursor.execute(query)
        
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error fetching users: {e}")
            return []

    def get_all_users(self):
        """Get all users - alias for backward compatibility"""
        return self.get_users_by_role()

    def get_faculty_users(self):
        """Get only faculty users"""
        return self.get_users_by_role('faculty')

    def get_student_users(self):
        """Get only student users"""
        return self.get_users_by_role('student')

    def get_admin_users(self):
        """Get only admin users"""
        return self.get_users_by_role('admin')

    def get_emails_from_csv_data(self, csv_file_path):
        """Extract email addresses from uploaded CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Debug: print column names and first few rows
            print(f"CSV columns: {list(df.columns)}")
            print(f"First 3 rows:\n{df.head(3)}")
            
            # Check if email column exists (case insensitive)
            email_column = None
            for col in df.columns:
                if col.lower().strip() in ['email', 'email_id', 'mail', 'e-mail']:
                    email_column = col
                    break
        
            if email_column:
                # Get unique, non-null email addresses
                emails = df[email_column].dropna().unique().tolist()
                valid_emails = []
                
                for email in emails:
                    email_str = str(email).strip()
                    # Basic email validation
                    if '@' in email_str and '.' in email_str and len(email_str) > 5:
                        valid_emails.append(email_str)
                
                print(f"Found {len(valid_emails)} valid emails: {valid_emails}")
                return valid_emails
            else:
                print(f"No email column found. Available columns: {list(df.columns)}")
                return []
                
        except Exception as e:
            print(f"Error reading emails from CSV: {e}")
            return []

    def get_student_emails_from_session_file(self, session_file_path):
        """Get student emails from session stored CSV file"""
        try:
            if session_file_path and os.path.exists(session_file_path):
                return self.get_emails_from_csv_data(session_file_path)
            else:
                print(f"Session file not found: {session_file_path}")
                return []
        except Exception as e:
            print(f"Error getting emails from session file: {e}")
            return []

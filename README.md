# Exam Seating Plan Generator - Complete Management System

## 📚 Project Overview

A comprehensive Python-based web application that automates fair exam seating arrangements with advanced user management, security features, and role-based access control. The system reads input data from CSV files and generates PDF seating plans while ensuring no cheating opportunities through intelligent constraint enforcement.

This enterprise-level solution features user authentication, dashboard management, administrative tools, email notifications, and comprehensive audit logging. Built with Flask web framework and MySQL database, it provides a professional interface using Bootstrap 5 and modern web technologies.

## 🏗️ Architecture

- **Backend**: Python Flask Framework with role-based routing
- **Database**: MySQL with comprehensive schema (8+ tables)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **PDF Generation**: xhtml2pdf with custom templates
- **Data Processing**: pandas and numpy for intelligent algorithms
- **Authentication**: bcrypt with secure session management
- **Email System**: SMTP integration for notifications

## ⚙️ Core Features

### 📝 Seating Plan Generation

- **CSV Input Support**: Upload student details and room layouts
- **Intelligent Constraints**:
  - No same-branch students side-by-side (left-right)
  - No same-branch students front-back
  - Global constraint enforcement across all rooms
- **Configurable Options**:
  - Students per desk (1-3)
  - Include/exclude detained students
  - Building/block selection
  - Branch-specific room assignments

### 👥 User Management & Authentication

- **Multi-Role System**: Admin, Faculty, Student roles
- **Secure Registration**: Email-based with validation
- **Login System**: Username/email authentication
- **Password Security**: bcrypt hashing, strength validation
- **Password Reset**: Secure token-based reset via email
- **Session Management**: Secure cookies, timeout protection

### 🔐 Security Features

- **Role-Based Access Control (RBAC)**
- **Session Hijacking Prevention**
- **Failed Login Attempt Logging**
- **Password Strength Requirements**
- **Secure Token Generation for Password Reset**
- **Input Validation and Sanitization**

### 📊 Dashboard & Management

- **Personalized Dashboard**: Role-specific views
- **Admin Panel**: User management, system monitoring
- **Activity Logging**: Comprehensive audit trail
- **System Statistics**: User counts, PDF generation stats
- **Real-time Monitoring**: System health indicators

### 📧 Communication & Notifications

- **Email Integration**: SMTP support for notifications
- **PDF Distribution**: Send seating plans to students/faculty
- **Password Reset Emails**: Secure reset links
- **System Notifications**: Important events and alerts

### 📄 Advanced PDF Features

- **Unlimited PDF Generation**
- **History Tracking**: All generated plans stored
- **Download Management**: Secure file access
- **Email Distribution**: Direct sending to stakeholders
- **Custom Templates**: Professional formatting

### ⚙️ System Configuration

- **Configurable Settings**:
  - Default students per desk
  - Session timeout periods
  - Password requirements
  - Email notification preferences
  - PDF retention policies
- **Building Management**: Multiple campus support
- **Branch Assignment**: Flexible room-branch mapping

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- MySQL Server 5.7 or higher
- Git

### Quick Start

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd sabi-s_project
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Database**

   - Create MySQL database: `seating_plan_db`
   - Update database credentials in `backend/models/database.py`:
     ```python
     self.host = 'localhost'
     self.database = 'seating_plan_db'
     self.user = 'your_username'
     self.password = 'your_password'
     ```

5. **Configure Email (Optional)**

   - Update email settings in `backend/models/database.py`:
     ```python
     self.smtp_server = 'smtp.gmail.com'
     self.smtp_port = 587
     self.email_username = 'your_email@gmail.com'
     self.email_password = 'your_app_password'
     ```

6. **Initialize Database**

   ```bash
   python backend/migrations/setup_database.py
   ```

7. **Run the Application**

   ```bash
   python backend/app.py
   ```

8. **Access the System**
   - Open browser to `http://localhost:5000`
   - Default admin credentials:
     - Username: `admin`
     - Password: `admin123`

## 📈 System Requirements

- **Python**: 3.8+ with Flask framework
- **Database**: MySQL 5.7+ or MariaDB
- **Memory**: 512MB RAM minimum
- **Storage**: 100MB for application + space for PDFs
- **Browser**: Modern browsers with JavaScript support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🆘 Support

- Create issues in the repository for bugs/features
- Contact development team for assistance
- Check documentation for common solutions

## 🔄 Version History

- **v2.0.0** - Complete system redesign with advanced features
- **v1.0.0** - Basic seating plan generation
- **v0.1.0** - Initial prototype

---

**Built with ❤️ for efficient exam management**

## 📁 Project Structure

```
sabi-s_project/
├── backend/                    # Backend application
│   ├── app.py                 # Main Flask application
│   ├── models/                # Database models and operations
│   │   └── database.py        # Database connection and models
│   ├── utils/                 # Utility modules
│   │   ├── seating.py         # Seating algorithm logic
│   │   └── pdf_export.py      # PDF generation utilities
│   └── migrations/            # Database migrations
│       ├── database_schema.sql
│       ├── database_migration.sql
│       ├── setup_database.py
│       ├── migrate_database.py
│       └── run_migration.py
├── frontend/                   # Frontend application
│   ├── templates/             # HTML templates
│   │   ├── dashboard.html     # Main dashboard
│   │   ├── login.html         # Login page
│   │   ├── signup.html        # Registration page
│   │   ├── admin.html         # Admin panel
│   │   ├── generate_plan.html # Plan generation
│   │   ├── seating_plan.html  # Plan display
│   │   ├── pdf_history.html   # History view
│   │   ├── settings.html      # User settings
│   │   └── *.html            # Other templates
│   └── static/               # Static assets
│       ├── css/
│       │   └── style.css     # Custom styles
│       └── js/               # JavaScript files
├── data/                      # Data storage
│   ├── uploads/              # Uploaded CSV files
│   ├── pdfs/                 # Generated PDF files
│   └── samples/              # Sample CSV files
│       ├── sample_students.csv
│       └── sample_rooms.csv
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .github/                  # GitHub configuration
└── .vscode/                  # VS Code configuration
```

## 🎯 Usage Guide

### For Administrators

1. **User Management**: Create and manage user accounts
2. **System Configuration**: Set global defaults and policies
3. **Monitor Activity**: View system logs and statistics
4. **Security Management**: Monitor login attempts and security

### For Faculty

1. **Generate Plans**: Create seating arrangements for exams
2. **Manage History**: Access previous seating plans
3. **Email Distribution**: Send plans to students
4. **Configure Preferences**: Set personal defaults

### For Students

1. **View Plans**: Access assigned seating arrangements
2. **Download PDFs**: Get personal copies of seating plans
3. **Account Management**: Update profile and preferences

## 📋 CSV File Formats

### Students CSV Format

```csv
Roll Number,Name,Branch,Section,Is Detained
2021001,John Doe,CSE,A,No
2021002,Jane Smith,ECE,B,No
2021003,Bob Johnson,CSE,A,Yes
```

### Rooms CSV Format

```csv
Room Number,Rows,Columns,Allowed Branches,Building
101,5,6,CSE;ECE,Main Building
102,4,7,CSE;ME,Main Building
201,6,5,ECE;EEE,Science Block
```

## 🔧 Configuration Options

### System Settings (Admin Only)

- **Students per Desk**: Default seating capacity
- **Include Detained**: Default policy for detained students
- **Session Timeout**: Security timeout in seconds
- **Password Requirements**: Minimum length and complexity
- **Email Notifications**: Enable/disable email features
- **PDF Retention**: How long to keep generated files

### User Preferences

- **Theme Selection**: Light/Dark/Auto themes
- **Language**: Multiple language support
- **Notification Preferences**: Email and system alerts
- **Default Building**: Preferred building for plans

## �️ Security Features

### Authentication & Authorization

- **Secure Password Hashing**: bcrypt implementation
- **Role-Based Access**: Granular permission system
- **Session Security**: Secure cookies and timeouts
- **Login Protection**: Failed attempt monitoring

### Data Protection

- **Input Validation**: Comprehensive form validation
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Output encoding and validation
- **CSRF Protection**: Token-based form protection

### Audit & Monitoring

- **Activity Logging**: All user actions tracked
- **Security Events**: Failed logins and suspicious activity
- **System Monitoring**: Real-time health checks
- **Error Logging**: Comprehensive error tracking

## 🚀 Deployment

### Production Considerations

1. **Environment Variables**: Use `.env` file for sensitive data
2. **Database Security**: Secure MySQL installation
3. **HTTPS**: Configure SSL/TLS certificates
4. **Email Service**: Set up reliable SMTP service
5. **Backup Strategy**: Regular database and file backups
6. **Monitoring**: Implement application monitoring

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "backend/app.py"]
```

## Database Schema

### Users Table

- `id` (INT, PRIMARY KEY)
- `username` (VARCHAR(50), UNIQUE)
- `email` (VARCHAR(100), UNIQUE)
- `password_hash` (VARCHAR(255))
- `full_name` (VARCHAR(100))
- `role` (ENUM: 'admin', 'faculty', 'student')
- `is_active` (BOOLEAN)
- `created_at` (TIMESTAMP)

### PDF History Table

- `id` (INT, PRIMARY KEY)
- `user_id` (INT, FOREIGN KEY)
- `filename` (VARCHAR(255))
- `file_path` (VARCHAR(500))
- `student_count` (INT)
- `room_count` (INT)
- `students_per_desk` (INT)
- `include_detained` (BOOLEAN)
- `created_at` (TIMESTAMP)

### Activity Logs Table

- `id` (INT, PRIMARY KEY)
- `user_id` (INT, FOREIGN KEY)
- `action` (VARCHAR(100))
- `details` (TEXT)
- `ip_address` (VARCHAR(45))
- `user_agent` (TEXT)
- `created_at` (TIMESTAMP)

## Troubleshooting

### Database Connection Issues

1. Ensure MySQL server is running
2. Check database credentials in `backend/models/database.py`
3. Verify database `seating_plan_db` exists
4. Run `python backend/migrations/setup_database.py` to recreate tables

### PDF Generation Issues

1. Check if `data/pdfs/` directory exists and is writable
2. Ensure xhtml2pdf is properly installed
3. Verify CSV files are properly formatted

### Authentication Issues

1. If database is unavailable, use fallback admin credentials
2. Check session configuration in `backend/app.py`
3. Clear browser cookies if login issues persist

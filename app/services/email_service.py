#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Email Service for Mercurio Edge.

This service handles sending emails for authentication, notifications, and other purposes.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@mercurioedge.com")
FROM_NAME = os.getenv("FROM_NAME", "Mercurio Edge")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"


class EmailService:
    """Service for sending emails."""

    def __init__(self):
        """Initialize email service."""
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.username = SMTP_USERNAME
        self.password = SMTP_PASSWORD
        self.from_email = FROM_EMAIL
        self.from_name = FROM_NAME
        self.enabled = EMAIL_ENABLED

    def send_email(self, to_email: str, subject: str, html_content: str, 
                  text_content: Optional[str] = None, cc: List[str] = None, 
                  bcc: List[str] = None) -> bool:
        """
        Send an email.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content of the email
            text_content: Plain text content of the email (fallback)
            cc: List of CC recipients
            bcc: List of BCC recipients

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(f"Email sending disabled. Would have sent email to {to_email} with subject: {subject}")
            return True

        if not all([self.smtp_server, self.username, self.password]):
            logger.error("Email configuration incomplete")
            return False

        try:
            # Create multipart message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            
            if cc:
                message["Cc"] = ", ".join(cc)
                
            if text_content is None:
                # Create a simple text version from HTML by removing tags
                text_content = html_content.replace("<br>", "\n").replace("<br/>", "\n").replace("<p>", "\n").replace("</p>", "\n")
                # Remove any remaining HTML tags
                import re
                text_content = re.sub(r'<[^>]+>', '', text_content)
            
            # Attach parts
            part1 = MIMEText(text_content, "plain")
            part2 = MIMEText(html_content, "html")
            message.attach(part1)
            message.attach(part2)
            
            # Calculate recipients for SMTP call
            recipients = [to_email]
            if cc:
                recipients.extend(cc)
            if bcc:
                recipients.extend(bcc)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, recipients, message.as_string())
                
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False

    def send_verification_email(self, email: str, verification_url: str) -> bool:
        """
        Send email verification email.
        
        Args:
            email: Recipient email address
            verification_url: URL for email verification

        Returns:
            True if email sent successfully, False otherwise
        """
        subject = "Verify Your Mercurio Edge Account"
        
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background-color: #0052cc;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }}
                    .button {{
                        display: inline-block;
                        background-color: #0052cc;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 4px;
                        margin: 20px 0;
                    }}
                    .footer {{
                        margin-top: 30px;
                        font-size: 12px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Mercurio Edge</h1>
                    </div>
                    <h2>Verify Your Email Address</h2>
                    <p>Thank you for creating an account with Mercurio Edge. To complete your registration, please verify your email address by clicking the button below:</p>
                    <p><a href="{verification_url}" class="button">Verify Email Address</a></p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p>{verification_url}</p>
                    <p>This link will expire in 3 days.</p>
                    <p>If you didn't create an account with Mercurio Edge, you can ignore this email.</p>
                    <div class="footer">
                        <p>Mercurio Edge - AI-Powered Trading Platform</p>
                        <p>&copy; 2025 Mercurio Systems. All rights reserved.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        return self.send_email(email, subject, html_content)

    def send_password_reset(self, email: str, reset_url: str) -> bool:
        """
        Send password reset email.
        
        Args:
            email: Recipient email address
            reset_url: URL for password reset

        Returns:
            True if email sent successfully, False otherwise
        """
        subject = "Reset Your Mercurio Edge Password"
        
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background-color: #0052cc;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }}
                    .button {{
                        display: inline-block;
                        background-color: #0052cc;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 4px;
                        margin: 20px 0;
                    }}
                    .footer {{
                        margin-top: 30px;
                        font-size: 12px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Mercurio Edge</h1>
                    </div>
                    <h2>Reset Your Password</h2>
                    <p>We received a request to reset your password for your Mercurio Edge account. Click the button below to set a new password:</p>
                    <p><a href="{reset_url}" class="button">Reset Password</a></p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p>{reset_url}</p>
                    <p>This link will expire in 1 hour.</p>
                    <p>If you didn't request a password reset, you can ignore this email and your password will remain unchanged.</p>
                    <div class="footer">
                        <p>Mercurio Edge - AI-Powered Trading Platform</p>
                        <p>&copy; 2025 Mercurio Systems. All rights reserved.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        return self.send_email(email, subject, html_content)

    def send_welcome_email(self, email: str, first_name: Optional[str] = None) -> bool:
        """
        Send welcome email to new users.
        
        Args:
            email: Recipient email address
            first_name: User's first name if available

        Returns:
            True if email sent successfully, False otherwise
        """
        greeting = f"Hi {first_name}" if first_name else "Welcome"
        subject = "Welcome to Mercurio Edge!"
        
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background-color: #0052cc;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }}
                    .button {{
                        display: inline-block;
                        background-color: #0052cc;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 4px;
                        margin: 20px 0;
                    }}
                    .footer {{
                        margin-top: 30px;
                        font-size: 12px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Mercurio Edge</h1>
                    </div>
                    <h2>{greeting} to Mercurio Edge!</h2>
                    <p>Thank you for joining Mercurio Edge, your AI-powered trading platform. We're excited to have you on board!</p>
                    <p>With Mercurio Edge, you can:</p>
                    <ul>
                        <li>Access advanced AI-powered trading strategies</li>
                        <li>Run backtests on historical data</li>
                        <li>Get real-time trading signals</li>
                        <li>Analyze market trends with cutting-edge tools</li>
                    </ul>
                    <p>To get started, log in to your account and explore our platform:</p>
                    <p><a href="https://app.mercurioedge.com/login" class="button">Go to Mercurio Edge</a></p>
                    <p>If you have any questions or need assistance, don't hesitate to contact our support team at support@mercurioedge.com.</p>
                    <p>Happy trading!</p>
                    <div class="footer">
                        <p>Mercurio Edge - AI-Powered Trading Platform</p>
                        <p>&copy; 2025 Mercurio Systems. All rights reserved.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        return self.send_email(email, subject, html_content)

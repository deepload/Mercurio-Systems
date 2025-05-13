"""Add subscription models

Revision ID: 001
Revises: 
Create Date: 2025-05-13

"""
from alembic import op
import sqlalchemy as sa
from enum import Enum

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create SubscriptionTier enum type
    op.execute("""
    CREATE TYPE subscription_tier AS ENUM ('FREE', 'STARTER', 'PRO', 'ELITE')
    """)
    
    # Create SubscriptionStatus enum type
    op.execute("""
    CREATE TYPE subscription_status AS ENUM ('active', 'trial', 'expired', 'cancelled', 'past_due')
    """)
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=True, default=False),
        sa.Column('email_verified_at', sa.DateTime(), nullable=True),
        sa.Column('first_name', sa.String(length=50), nullable=True),
        sa.Column('last_name', sa.String(length=50), nullable=True),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('profile_image_url', sa.String(length=255), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('risk_profile', sa.String(length=20), nullable=True),
        sa.Column('investment_goals', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('now()')),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('tier', sa.Enum('FREE', 'STARTER', 'PRO', 'ELITE', name='subscription_tier'), nullable=True, default='FREE'),
        sa.Column('status', sa.Enum('active', 'trial', 'expired', 'cancelled', 'past_due', name='subscription_status'), nullable=True, default='active'),
        sa.Column('is_trial', sa.Boolean(), nullable=True, default=False),
        sa.Column('trial_started_at', sa.DateTime(), nullable=True),
        sa.Column('trial_ends_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('now()')),
        sa.Column('current_period_start', sa.DateTime(), nullable=True),
        sa.Column('current_period_end', sa.DateTime(), nullable=True),
        sa.Column('external_subscription_id', sa.String(length=255), nullable=True),
        sa.Column('payment_method_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_subscriptions_id'), 'subscriptions', ['id'], unique=False)
    op.create_index(op.f('ix_subscriptions_status'), 'subscriptions', ['status'], unique=False)
    op.create_index(op.f('ix_subscriptions_tier'), 'subscriptions', ['tier'], unique=False)
    op.create_index(op.f('ix_subscriptions_user_id'), 'subscriptions', ['user_id'], unique=False)
    
    # Create subscription_payments table
    op.create_table(
        'subscription_payments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=True),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=True, server_default='USD'),
        sa.Column('external_payment_id', sa.String(length=255), nullable=True),
        sa.Column('payment_method', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('payment_date', sa.DateTime(), nullable=True, server_default=sa.text('now()')),
        sa.Column('period_start', sa.DateTime(), nullable=True),
        sa.Column('period_end', sa.DateTime(), nullable=True),
        sa.Column('receipt_url', sa.String(length=255), nullable=True),
        sa.Column('invoice_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['subscription_id'], ['subscriptions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_subscription_payments_id'), 'subscription_payments', ['id'], unique=False)
    op.create_index(op.f('ix_subscription_payments_status'), 'subscription_payments', ['status'], unique=False)
    op.create_index(op.f('ix_subscription_payments_subscription_id'), 'subscription_payments', ['subscription_id'], unique=False)


def downgrade() -> None:
    # Drop subscription_payments table
    op.drop_index(op.f('ix_subscription_payments_subscription_id'), table_name='subscription_payments')
    op.drop_index(op.f('ix_subscription_payments_status'), table_name='subscription_payments')
    op.drop_index(op.f('ix_subscription_payments_id'), table_name='subscription_payments')
    op.drop_table('subscription_payments')
    
    # Drop subscriptions table
    op.drop_index(op.f('ix_subscriptions_user_id'), table_name='subscriptions')
    op.drop_index(op.f('ix_subscriptions_tier'), table_name='subscriptions')
    op.drop_index(op.f('ix_subscriptions_status'), table_name='subscriptions')
    op.drop_index(op.f('ix_subscriptions_id'), table_name='subscriptions')
    op.drop_table('subscriptions')
    
    # Drop users table
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    
    # Drop enum types
    op.execute("DROP TYPE subscription_status")
    op.execute("DROP TYPE subscription_tier")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run subscription tests directly without the global pytest configuration.
This is a workaround until the global test configuration is fixed.

Usage:
    python run_tests.py [options]

Options:
    --all           Run all tests
    --config        Run config tests
    --adapter       Run adapter tests
    --service       Run service tests
    --middleware    Run middleware tests
    --routes        Run routes tests
    --stripe        Run Stripe payment service tests
    --integrated    Run integrated subscription service tests with Stripe
    --help          Show help message

Example:
    python run_tests.py --config --adapter  # Run only config and adapter tests
    python run_tests.py --stripe --integrated  # Run Stripe integration tests
    python run_tests.py --all               # Run all tests
"""

import unittest
import sys
import os
import argparse
from unittest.mock import patch

# No need to import the async_test utility here
# It's now imported directly in the test files

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test modules, using try/except to handle potential import errors
try:
    from test_subscription_config import TestSubscriptionConfig
    config_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import config tests: {e}")
    config_tests_available = False

try:
    from test_subscription_adapter import TestSubscriptionAdapter
    adapter_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import adapter tests: {e}")
    adapter_tests_available = False

try:
    from test_subscription_service import TestSubscriptionService
    service_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import service tests: {e}")
    service_tests_available = False

try:
    from test_subscription_middleware import TestSubscriptionMiddleware
    middleware_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import middleware tests: {e}")
    middleware_tests_available = False

try:
    from test_subscription_routes import TestSubscriptionRoutes
    routes_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import routes tests: {e}")
    routes_tests_available = False

try:
    from test_stripe_service import TestStripeService
    stripe_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import Stripe service tests: {e}")
    stripe_tests_available = False

try:
    from test_integrated_subscription_service import TestIntegratedSubscriptionService
    integrated_tests_available = True
except ImportError as e:
    print(f"Warning: Could not import integrated subscription service tests: {e}")
    integrated_tests_available = False

def run_test_suite(run_config=False, run_adapter=False, run_service=False, 
                run_middleware=False, run_routes=False, run_stripe=False,
                run_integrated=False, run_all=False):
    """
    Run the specified test suites.
    
    This function handles both regular tests and async tests. For the async tests,
    it uses the asyncio event loop to properly support the async/await pattern.
    """
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Set all flags to True if run_all is specified
    if run_all:
        run_config = run_adapter = run_service = run_middleware = run_routes = run_stripe = run_integrated = True
    
    # If no specific options were provided, run the reliable tests by default
    if not any([run_config, run_adapter, run_service, run_middleware, run_routes, run_stripe, run_integrated]):
        run_config = run_adapter = True
        print("No test options specified. Running only config and adapter tests by default.")
    
    # Add specified test suites with proper error handling
    if run_config and config_tests_available:
        print("Adding subscription config tests")
        test_suite.addTest(unittest.makeSuite(TestSubscriptionConfig))
    
    if run_adapter and adapter_tests_available:
        print("Adding subscription adapter tests")
        test_suite.addTest(unittest.makeSuite(TestSubscriptionAdapter))
    
    if run_service and service_tests_available:
        print("Adding subscription service tests")
        # Set up any special mocks needed for service tests
        test_suite.addTest(unittest.makeSuite(TestSubscriptionService))
    
    if run_middleware and middleware_tests_available:
        print("Adding subscription middleware tests")
        # Set up any special mocks needed for middleware tests
        test_suite.addTest(unittest.makeSuite(TestSubscriptionMiddleware))
    
    if run_routes and routes_tests_available:
        print("Adding subscription routes tests")
        with patch('stripe.WebhookSignature.verify', return_value=True):
            test_suite.addTest(unittest.makeSuite(TestSubscriptionRoutes))
            
    if run_stripe and stripe_tests_available:
        print("Adding Stripe payment service tests")
        test_suite.addTest(unittest.makeSuite(TestStripeService))
        
    if run_integrated and integrated_tests_available:
        print("Adding integrated subscription service tests")
        test_suite.addTest(unittest.makeSuite(TestIntegratedSubscriptionService))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Our async tests now use the async_test decorator to handle the event loop
    if run_stripe or run_integrated:
        print("Running tests with async support enabled")
    
    # Run the tests
    result = runner.run(test_suite)
    
    print("\nNote: Some tests may fail if they require external dependencies")
    print("or if their mock objects are not properly configured.")
    
    return result.wasSuccessful()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run subscription tests")
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--config', action='store_true', help='Run config tests')
    parser.add_argument('--adapter', action='store_true', help='Run adapter tests')
    parser.add_argument('--service', action='store_true', help='Run service tests')
    parser.add_argument('--middleware', action='store_true', help='Run middleware tests')
    parser.add_argument('--routes', action='store_true', help='Run routes tests')
    parser.add_argument('--stripe', action='store_true', help='Run Stripe payment service tests')
    parser.add_argument('--integrated', action='store_true', help='Run integrated subscription service tests')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup environment variables that might be needed by tests
    os.environ.setdefault('ALPACA_API_KEY', 'test_key')
    os.environ.setdefault('ALPACA_API_SECRET', 'test_secret')
    os.environ.setdefault('POLYGON_API_KEY', 'test_key')
    
    # Run with the specified options
    print("\nMercurio Subscription Tests\n" + "-" * 30)
    success = run_test_suite(
        run_config=args.config, 
        run_adapter=args.adapter, 
        run_service=args.service,
        run_middleware=args.middleware, 
        run_routes=args.routes,
        run_stripe=args.stripe,
        run_integrated=args.integrated,
        run_all=args.all
    )
    
    # Exit with appropriate code
    if not success:
        print("\nTests FAILED")
        sys.exit(1)
    else:
        print("\nTests PASSED")
        sys.exit(0)

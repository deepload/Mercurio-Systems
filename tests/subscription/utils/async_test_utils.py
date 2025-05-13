#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for testing async code with unittest.
"""

import asyncio
import functools
import sys

def async_test(func):
    """
    Decorator for async test methods to make them work with unittest.
    
    Usage:
    ```
    class MyTest(unittest.TestCase):
        @async_test
        async def test_async_function(self):
            # Your async test code here
            result = await some_async_function()
            self.assertEqual(result, expected_value)
    ```
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a new event loop for each test
        policy = asyncio.WindowsSelectorEventLoopPolicy() if sys.platform == "win32" else asyncio.DefaultEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the test function in the loop
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            # Close the loop to avoid resource warnings
            loop.close()
            
    return wrapper

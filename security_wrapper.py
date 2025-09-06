
"""Security wrapper injected by Caminu for customer agent safety"""

import resource
import signal
import os
import sys
from functools import wraps
from typing import Any, Callable

# Set resource limits
try:
    # CPU time limit: 5 minutes per request
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    
    # Memory limit: 512MB
    resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    
    # File size limit: 10MB
    resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
    
    # Number of processes: 1 (no forking)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    
except Exception:
    pass  # Some limits might not be available

# Request timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Request timeout: exceeded 30 seconds")

def secure_request_wrapper(timeout_seconds: int = 30):
    """Decorator to add security and timeout to agent requests"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set request timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                # Execute the original function
                result = func(*args, **kwargs)
                return result
            finally:
                # Clear the alarm
                signal.alarm(0)
        
        return wrapper
    return decorator

# Apply security wrapper to main agent function
# This will be automatically applied to the agent's main function

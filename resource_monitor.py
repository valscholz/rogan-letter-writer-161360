
"""Resource monitoring injected by Caminu"""

import psutil
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor resource usage and send alerts if limits are exceeded"""
    
    def __init__(self, customer_id: str = "c6b07ad4-8ce0-497d-82eb-c52eb28b7157"):
        self.customer_id = customer_id
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
    
    def check_resources(self):
        """Check current resource usage"""
        process = psutil.Process()
        
        # Check memory usage
        current_memory = process.memory_info().rss
        memory_mb = current_memory / (1024 * 1024)
        
        if memory_mb > 400:  # 400MB warning threshold
            logger.warning(f"High memory usage for customer {self.customer_id}: {memory_mb:.1f}MB")
        
        # Check CPU usage
        cpu_percent = process.cpu_percent()
        if cpu_percent > 80:
            logger.warning(f"High CPU usage for customer {self.customer_id}: {cpu_percent:.1f}%")
        
        # Check runtime
        runtime = time.time() - self.start_time
        if runtime > 300:  # 5 minutes
            logger.warning(f"Long running request for customer {self.customer_id}: {runtime:.1f}s")
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "runtime_seconds": runtime,
            "timestamp": datetime.now().isoformat()
        }

# Global monitor instance
resource_monitor = ResourceMonitor()

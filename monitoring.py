from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import psutil
import time
import threading

# Metrics definitions
REQUEST_COUNT = Counter('ml_model_requests_total', 
                       'Total number of requests served', 
                       ['method', 'endpoint', 'status'])

REQUEST_DURATION = Histogram('ml_model_request_duration_seconds',
                            'Request duration in seconds',
                            ['method', 'endpoint'])

PREDICTION_COUNT = Counter('ml_model_predictions_total',
                          'Total number of predictions made')

PREDICTION_PROBABILITY = Histogram('ml_model_prediction_probability',
                                  'Prediction probability distribution')

RESPONSE_TIME = Histogram('ml_model_response_time_seconds',
                         'Response time for predictions')

# System metrics
CPU_USAGE = Gauge('ml_model_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('ml_model_memory_usage_mb', 'Memory usage in MB')
ACTIVE_REQUESTS = Gauge('ml_model_active_requests', 'Number of active requests')

class SystemMonitor:
    def __init__(self):
        self.stop_monitor = False
    
    def collect_system_metrics(self):
        while not self.stop_monitor:
            try:
                # CPU usage
                CPU_USAGE.set(psutil.cpu_percent())
                
                # Memory usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.set(memory.used / 1024 / 1024)  # Convert to MB
                
                time.sleep(5)  # Collect every 5 seconds
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
    
    def start(self):
        self.monitor_thread = threading.Thread(target=self.collect_system_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        self.stop_monitor = True

# Global system monitor instance
system_monitor = SystemMonitor()
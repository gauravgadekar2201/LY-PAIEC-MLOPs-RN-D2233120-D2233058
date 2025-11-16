import requests
import time
import random
import threading
import json

BASE_URL = "http://localhost:8000"

def generate_random_features():
    return [random.uniform(-2, 2) for _ in range(20)]

def make_prediction():
    try:
        features = generate_random_features()
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=features,
            headers={"Content-Type": "application/json"}
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction: {result['prediction']}, "
                  f"Response Time: {result['response_time']:.3f}s")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def load_test():
    print("Starting load test...")
    threads = []
    
    # Create 50 concurrent requests
    for i in range(50):
        thread = threading.Thread(target=make_prediction)
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Stagger requests
    
    for thread in threads:
        thread.join()
    
    print("Load test completed!")

def check_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

def get_metrics():
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print("Metrics endpoint is working!")
        # Print some key metrics
        lines = response.text.split('\n')
        for line in lines:
            if any(key in line for key in ['requests_total', 'cpu_usage', 'memory_usage', 'predictions_total']):
                print(line)
    except Exception as e:
        print(f"Metrics check failed: {e}")

if __name__ == "__main__":
    print("Testing ML Model Monitoring Setup...")
    
    # Wait for app to start
    time.sleep(5)
    
    check_health()
    print("\n" + "="*50)
    
    get_metrics()
    print("\n" + "="*50)
    
    # Make some individual predictions
    print("Making individual predictions:")
    for i in range(5):
        make_prediction()
        time.sleep(1)
    
    print("\n" + "="*50)
    
    # Run load test
    load_test()
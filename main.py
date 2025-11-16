from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest
import time
import logging
from typing import List

from .model import MLModel
from .monitoring import (
    REQUEST_COUNT, REQUEST_DURATION, PREDICTION_COUNT,
    PREDICTION_PROBABILITY, RESPONSE_TIME, ACTIVE_REQUESTS,
    system_monitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model Monitoring Demo", version="1.0.0")

# Initialize model and monitoring
ml_model = MLModel()
system_monitor.start()

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        
        # Record request metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise e
        
    finally:
        process_time = time.time() - start_time
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        ACTIVE_REQUESTS.dec()

@app.get("/")
async def root():
    return {"message": "ML Model Monitoring API", "status": "healthy"}

@app.post("/predict")
async def predict(features: List[float]):
    start_time = time.time()
    
    try:
        if len(features) != 20:
            return JSONResponse(
                status_code=400,
                content={"error": "Exactly 20 features required"}
            )
        
        # Make prediction
        result = ml_model.predict(features)
        
        # Record prediction metrics
        PREDICTION_COUNT.inc()
        PREDICTION_PROBABILITY.observe(result["probability"])
        
        response_time = time.time() - start_time
        RESPONSE_TIME.observe(response_time)
        
        logger.info(f"Prediction made: {result['prediction']}, "
                   f"probability: {result['probability']:.3f}, "
                   f"response_time: {response_time:.3f}s")
        
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "response_time": response_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed"}
        )

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024
        }
    }

@app.on_event("shutdown")
def shutdown_event():
    system_monitor.stop()
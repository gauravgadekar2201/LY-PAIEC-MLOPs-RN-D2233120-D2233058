#!/bin/bash

echo "Starting ML Model Monitoring Setup..."

# Build and start services
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo "Waiting for services to start..."
sleep 30

echo "Services are running!"
echo "ML Model API: http://localhost:8000"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"

echo ""
echo "Running load test..."
python test_script.py

echo ""
echo "Setup complete! Access the monitoring dashboards:"
echo "- Grafana: http://localhost:3000"
echo "- Prometheus: http://localhost:9090"
echo "- ML API: http://localhost:8000"
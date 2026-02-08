#!/usr/bin/env bash

set -e

FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501

echo "Starting FastAPI on port ${FASTAPI_PORT}..."
uvicorn base:app \
  --host ${FASTAPI_HOST} \
  --port ${FASTAPI_PORT} \
  --reload &

FASTAPI_PID=$!

echo "Starting Streamlit on port ${STREAMLIT_PORT}..."
streamlit run main.py \
  --server.port ${STREAMLIT_PORT} \
  --server.address 0.0.0.0 &

STREAMLIT_PID=$!

echo ""
echo "FastAPI     → http://localhost:${FASTAPI_PORT}"
echo "Streamlit   → http://localhost:${STREAMLIT_PORT}"
echo ""
echo "Press Ctrl+C to stop both"

# Clean shutdown on Ctrl+C
trap "echo 'Stopping...'; kill $FASTAPI_PID $STREAMLIT_PID" SIGINT SIGTERM

wait

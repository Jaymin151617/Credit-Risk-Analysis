#!/usr/bin/env bash

###############################################################################
# run.sh — Development Startup Script
#
# Description
# -----------
# Starts both:
#   1. FastAPI backend (via uvicorn)
#   2. Streamlit frontend
#
# This script is intended for LOCAL DEVELOPMENT ONLY.
# It enables hot-reload for FastAPI and runs both services concurrently.
#
# Usage
# -----
#   chmod +x run.sh
#   ./run.sh
#
# Notes
# -----
# - Uses background processes (&) to run both services.
# - Implements clean shutdown handling (Ctrl+C).
# - Not intended for production deployment.
###############################################################################

# Exit immediately if any command fails
# Prevents silent failures during startup
set -e


###############################################################################
# Configuration
###############################################################################

# Host binding for FastAPI (0.0.0.0 allows access from other devices)
FASTAPI_HOST=0.0.0.0

# Port for backend API
FASTAPI_PORT=8000

# Port for Streamlit frontend
STREAMLIT_PORT=8501


###############################################################################
# Start FastAPI (Backend)
###############################################################################

echo "Starting FastAPI on port ${FASTAPI_PORT}..."

# Launch uvicorn with auto-reload enabled (dev-only feature)
# --reload watches files and restarts server on changes
uvicorn app.api:app \
  --host ${FASTAPI_HOST} \
  --port ${FASTAPI_PORT} \
  --reload &

# Capture process ID for shutdown handling
FASTAPI_PID=$!


###############################################################################
# Start Streamlit (Frontend)
###############################################################################

echo "Starting Streamlit on port ${STREAMLIT_PORT}..."

# Run Streamlit app bound to all interfaces
# This allows access via localhost or network IP
streamlit run app/main.py \
  --server.port ${STREAMLIT_PORT} \
  --server.address 0.0.0.0 &

# Capture process ID for shutdown handling
STREAMLIT_PID=$!


###############################################################################
# Graceful Shutdown Handling
###############################################################################

# On Ctrl+C (SIGINT) or termination (SIGTERM):
# - Print message
# - Kill both background processes
trap "echo 'Stopping services...'; kill $FASTAPI_PID $STREAMLIT_PID" SIGINT SIGTERM


###############################################################################
# Keep Script Running
###############################################################################

# Wait for background processes to exit.
# Prevents script from terminating immediately.
wait

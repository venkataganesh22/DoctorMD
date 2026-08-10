#!/bin/bash

# Start FastAPI backend internally
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# Give FastAPI a moment to start
sleep 3

# Start Streamlit as the public application
streamlit run frontend/app.py \
    --server.port $PORT \
    --server.address 0.0.0.0

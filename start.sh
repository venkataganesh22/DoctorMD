#!/bin/bash

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

sleep 3

streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0

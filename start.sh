#!/bin/bash
# Sentinel – Start Script (macOS / Linux)
# Runs sentinel_detect.py and sentinel.py concurrently.
# Usage: 
# chmod +x start.sh
# ./start.sh

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

echo ""
echo "Sentinel Startup"
echo "=================="
echo ""

# VIRTUAL ENVIRONMENT
if [ ! -d "$VENV_DIR" ]; then
    echo "[sentinel] No virtual environment found. Creating one..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[sentinel] ERROR: Failed to create virtual environment. Is python3 installed?"
        exit 1
    fi
    echo "[sentinel] Virtual environment created at $VENV_DIR"
 
    echo "[sentinel] Installing requirements..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" -q
    if [ $? -ne 0 ]; then
        echo "[sentinel] ERROR: Failed to install requirements."
        exit 1
    fi
    echo "[sentinel] Requirements installed."
else
    echo "[sentinel] Virtual environment found at $VENV_DIR"
fi
 
# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "[sentinel] Virtual environment activated."
echo ""

# .ENV CHECK
if [ ! -f ".env" ]; then
    echo "! WARNING !"
    echo "No .env file found in this directory."
    echo ""
    echo "Create one by running:"
    echo "cp .env.example .env"
    echo "Then fill in your credentials and rerun this script."
    echo ""
    deactivate 2>/dev/null
    exit 1
fi
echo "[sentinel] .env file found."

# SETTINGS CHECK

# secrets folder; for the credentials
if [ ! -d "secrets" ]; then
    mkdir -p secrets
    echo "[sentinel] secrets/ folder created."
else
    echo "[sentinel] secrets/ folder found."
fi

# settings.json
if [ ! -f "settings.json" ]; then
    echo "[sentinel] settings.json not found — creating with defaults..."
    cat > settings.json << 'EOF'
{
  "capture_interval": 10,
  "confidence_min": 0.55
}
EOF
    echo "[sentinel] settings.json created."
else
    echo "[sentinel] settings.json found."
fi
echo ""

# CLEANUP
# Kill both processes cleanly on Ctrl+C
cleanup() {
    echo ""
    echo "[sentinel] Shutting down all services..."
 
    # Kill all child processes
    [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null
    [ -n "$DETECT_PID"    ] && kill "$DETECT_PID"    2>/dev/null
    [ -n "$API_PID"       ] && kill "$API_PID"       2>/dev/null
 
    # Wait briefly for them to actually stop
    sleep 1
 
    # Force kill anything still running
    [ -n "$STREAMLIT_PID" ] && kill -9 "$STREAMLIT_PID" 2>/dev/null
    [ -n "$DETECT_PID"    ] && kill -9 "$DETECT_PID"    2>/dev/null
    [ -n "$API_PID"       ] && kill -9 "$API_PID"       2>/dev/null
 
    # Deactivate virtual environment
    deactivate 2>/dev/null
 
    echo "[sentinel] All services stopped. Goodbye."
    exit 0
}
 
# Trap Ctrl+C and termination signals — only bind once
trap cleanup SIGINT SIGTERM

# SERVICES
# Start detection pipeline in background
echo "[sentinel] Starting API (port 8000)..."
uvicorn sentinel_api:app --host 0.0.0.0 --port 8000 --log-level warning &
API_PID=$!
sleep 2
 
echo "[sentinel] Starting detection pipeline..."
python3 sentinel_detect.py &
DETECT_PID=$!
sleep 1
 
echo "[sentinel] Starting Streamlit dashboard..."
streamlit run sentinel.py --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo ""
echo "[sentinel] Both services running."
echo "[sentinel] Press Ctrl+C to stop everything."
echo ""

# Wait for either process to exit
wait $API_PID $DETECT_PID $STREAMLIT_PID
cleanup
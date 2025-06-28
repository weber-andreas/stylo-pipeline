#!/bin/bash

# --- Configuration ---
PORT=8000
CONDA_ENV="stylo2"

# --- Input Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <username> <hostname> <ssh_key_location>"
    exit 1
fi

USERNAME=$1
HOST=$2
SSH_KEY=$3

# --- Script ---
echo "Running on node: $(hostname)"

# Activate Conda Environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Error: Could not activate conda environment '$CONDA_ENV'"
    exit 1
fi
echo "Activated conda environment"

# Port Forwarding
echo "Attempting to forward Port:"
echo "  Port: $PORT"
echo "  Username: $USERNAME"
echo "  Host: $HOST"
echo "  SSH key location: $SSH_KEY"

ssh -N -f -R "$PORT":0.0.0.0:"$PORT" "$USERNAME"@"$HOST" -p 58022 -i "$SSH_KEY"
if [ $? -eq 0 ]; then
    echo "Successfully forwarded port to $HOST:$PORT"
else
    echo "Error: Port forwarding failed. Please check your ssh connection and credentials."
    exit 1
fi

# Start Server
echo "Starting server now"
python3 ./src/server/server.py --host 0.0.0.0 --port $PORT



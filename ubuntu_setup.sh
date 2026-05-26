#!/bin/bash
set -e

echo "=========================================="
echo "Starting AI Voice Bot Ubuntu Preparation..."
echo "=========================================="

sudo apt update -y
sudo apt install -y python3-pip python3-venv ffmpeg libasound2-dev build-essential curl redis-server postgresql postgresql-contrib unzip wget

sudo systemctl enable redis-server
sudo systemctl start redis-server
echo "[SUCCESS] Redis Server is running!"

sudo systemctl enable postgresql
sudo systemctl start postgresql
echo "[SUCCESS] PostgreSQL Server is running!"

DB_NAME="voicebot_db"
DB_USER="voicebot_user"
DB_PASS="VoiceBotPass99!!"

sudo -i -u postgres psql -c "CREATE DATABASE $DB_NAME;" || true
sudo -i -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" || true
sudo -i -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" || true
echo "[SUCCESS] PostgreSQL Database '$DB_NAME' created and secured!"

# Set up project env
PROJECT_DIR="/opt/voicebot"
cd $PROJECT_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn websockets httpx asyncpg redis edge-tts standard-aifc cryptography vosk pydantic python-dotenv

echo "[SUCCESS] Setup finished!"

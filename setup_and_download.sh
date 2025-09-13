#!/bin/bash
# Setup script for VOO/QQQ Strategy Project
# This script installs dependencies and downloads ETF data

echo "=========================================="
echo "VOO/QQQ Strategy - 安裝與下載腳本"
echo "=========================================="
echo ""

# Step 1: Update package list
echo "步驟 1: 更新套件列表..."
sudo apt update

# Step 2: Install pip if not already installed
echo ""
echo "步驟 2: 安裝 Python pip..."
sudo apt install -y python3-pip

# Step 3: Install Python packages
echo ""
echo "步驟 3: 安裝 Python 套件..."
pip3 install --upgrade pip
pip3 install yfinance pandas numpy python-dateutil matplotlib seaborn openpyxl

# Step 4: Verify installations
echo ""
echo "步驟 4: 驗證安裝..."
python3 -c "import yfinance; print('✅ yfinance 安裝成功')"
python3 -c "import pandas; print('✅ pandas 安裝成功')"
python3 -c "import numpy; print('✅ numpy 安裝成功')"

# Step 5: Download VOO data
echo ""
echo "步驟 5: 下載 VOO 歷史資料..."
echo "=========================================="
cd /mnt/c/Jane/ClaudeCode/20250913_VOOnQQQ
python3 download_voo.py

echo ""
echo "=========================================="
echo "✅ 安裝與下載完成！"
echo "=========================================="
echo ""
echo "資料檔案位置："
echo "  - data/raw/VOO_latest.csv"
echo ""
ls -la data/raw/VOO*.csv 2>/dev/null || echo "（等待資料下載完成）"
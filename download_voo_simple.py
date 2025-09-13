#!/usr/bin/env python3
"""
Simple VOO data downloader using urllib (no external dependencies)
This downloads VOO data directly from Yahoo Finance API
"""

import urllib.request
import json
import csv
from datetime import datetime, timezone
import os
from pathlib import Path

def download_voo_simple():
    """Download VOO data using Yahoo Finance URL without yfinance"""

    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VOO 歷史資料下載 (簡易版)")
    print("="*60)

    # Convert dates to timestamps
    # Start: Jan 1, 2005
    start_timestamp = int(datetime(2005, 1, 1, tzinfo=timezone.utc).timestamp())
    # End: Current date
    end_timestamp = int(datetime.now(timezone.utc).timestamp())

    # Yahoo Finance download URL
    ticker = "VOO"
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    url += f"?period1={start_timestamp}"
    url += f"&period2={end_timestamp}"
    url += "&interval=1d"
    url += "&events=history"
    url += "&includeAdjustedClose=true"

    print(f"下載 {ticker} 資料...")
    print(f"從 2005-01-01 到今天")

    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        request = urllib.request.Request(url, headers=headers)

        # Download data
        with urllib.request.urlopen(request) as response:
            data = response.read().decode('utf-8')

        # Save to CSV file
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"VOO_historical_data_{timestamp}.csv"
        filepath = data_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)

        print(f"✅ 資料已儲存到: {filepath}")

        # Also save as latest version
        latest_filepath = data_dir / "VOO_latest.csv"
        with open(latest_filepath, 'w', encoding='utf-8') as f:
            f.write(data)

        print(f"✅ 最新版本已儲存: {latest_filepath}")

        # Count rows
        lines = data.strip().split('\n')
        print(f"✅ 下載完成！共 {len(lines)-1} 筆資料")

        # Show last 5 rows
        print("\n最近 5 筆資料:")
        print(lines[0])  # Header
        for line in lines[-5:]:
            print(line)

        return True

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP 錯誤: {e.code} - {e.reason}")
        print("可能需要安裝 yfinance 套件來下載資料")
        return False
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_voo_simple()

    if not success:
        print("\n" + "="*60)
        print("如果下載失敗，請執行以下指令安裝必要套件：")
        print("1. sudo apt update")
        print("2. sudo apt install -y python3-pip")
        print("3. pip3 install yfinance pandas")
        print("4. python3 download_voo.py")
        print("="*60)
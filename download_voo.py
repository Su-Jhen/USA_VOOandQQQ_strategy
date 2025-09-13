#!/usr/bin/env python3
"""
Quick script to download VOO historical data
Run from project root: python download_voo.py
"""

import sys
from pathlib import Path

# Add src/main/python to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from services.data_downloader import ETFDataDownloader

if __name__ == "__main__":
    print("開始下載 VOO 歷史資料...")

    # Create downloader
    downloader = ETFDataDownloader()

    # Download VOO data from 2005 to 2025
    voo_data = downloader.download_and_save_voo(
        start_date="2005-01-01",
        end_date="2025-12-31"
    )

    if voo_data is not None:
        print(f"\n✅ 成功下載並儲存 VOO 資料!")
        print(f"檔案位置: data/raw/VOO_latest.csv")
    else:
        print("\n❌ 下載失敗，請檢查網路連線或稍後再試")
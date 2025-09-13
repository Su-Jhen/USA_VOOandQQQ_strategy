#!/usr/bin/env python3
"""
Download complete historical data for VOO, QQQ, and SPY
"""

import sys
from pathlib import Path

# Add src/main/python to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from services.etf_full_downloader import ETFFullDownloader

if __name__ == "__main__":
    print("開始下載完整歷史資料...")
    print("="*70)

    downloader = ETFFullDownloader()
    results = downloader.download_all_etfs()

    # Also download SPY for 2005-2010 S&P 500 data
    print("\n自動下載 SPY 作為 2005-2010 S&P 500 資料來源...")
    spy_data = downloader.download_spy_as_alternative()

    print("\n✅ 所有資料下載完成！")
    print("\n檔案位置:")
    print("  - data/raw/VOO_complete.csv (2010-2025)")
    print("  - data/raw/QQQ_complete.csv (1999-2025)")
    print("  - data/raw/SPY_complete.csv (1993-2025)")
    print("\n註：SPY 可用於分析 2005-2010 的 S&P 500 表現")
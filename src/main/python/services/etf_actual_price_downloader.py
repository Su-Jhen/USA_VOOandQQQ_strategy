"""
ETF Actual Historical Price Downloader
Downloads actual historical prices (non-adjusted) for ETFs
保存實際歷史價格，不含股息調整
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


class ETFActualPriceDownloader:
    """Downloads actual historical prices (non-adjusted) for ETFs"""

    # ETF inception dates (first trading day)
    ETF_INFO = {
        'VOO': {
            'name': 'Vanguard S&P 500 ETF',
            'inception_date': '2010-09-07',
            'description': '追蹤 S&P 500 指數'
        },
        'QQQ': {
            'name': 'Invesco QQQ Trust',
            'inception_date': '1999-03-10',
            'description': '追蹤 NASDAQ-100 指數'
        },
        'SPY': {
            'name': 'SPDR S&P 500 ETF Trust',
            'inception_date': '1993-01-29',
            'description': '最早的 S&P 500 ETF'
        },
        'UPRO': {
            'name': 'ProShares UltraPro S&P 500',
            'inception_date': '2009-06-23',
            'description': '3倍槓桿 S&P 500 ETF'
        }
    }

    def __init__(self, data_dir=None):
        """Initialize downloader"""
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            data_dir = project_root / "data" / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_actual_prices(self, ticker, save_format='csv'):
        """Download actual historical prices (non-adjusted) for an ETF"""
        print(f"\n{'='*60}")
        print(f"下載 {ticker} 實際歷史價格（未調整）")
        print(f"{'='*60}")

        try:
            # Create ticker object
            etf = yf.Ticker(ticker)

            # Download all available historical data with auto_adjust=False
            print(f"正在下載 {ticker} 的實際歷史價格...")

            # Get the raw data without adjustments
            data = etf.history(period="max", auto_adjust=False, back_adjust=False)

            if data.empty:
                print(f"❌ 無法下載 {ticker} 的資料")
                return None

            # Reset index to make Date a column
            data.reset_index(inplace=True)

            # Select only the actual price columns (not adjusted)
            # Keep: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
            # Remove: Adj Close (if exists)
            columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            if 'Adj Close' in data.columns:
                print("📊 使用實際收盤價 (Close)，不使用調整後收盤價 (Adj Close)")

            # Filter columns
            available_columns = [col for col in columns_to_keep if col in data.columns]
            data = data[available_columns]

            # Add ticker column
            data['Ticker'] = ticker

            # Add a note column to indicate this is actual price
            data['Price_Type'] = 'Actual'

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if save_format == 'csv':
                # Save timestamped version
                filename = f"{ticker}_actual_prices_{timestamp}.csv"
                filepath = self.data_dir / filename
                data.to_csv(filepath, index=False)
                print(f"✅ 實際歷史價格已儲存: {filepath}")

                # Save latest version
                latest_filename = f"{ticker}_actual.csv"
                latest_filepath = self.data_dir / latest_filename
                data.to_csv(latest_filepath, index=False)
                print(f"✅ 最新版本已儲存: {latest_filepath}")

            # Display summary
            print(f"\n下載完成統計:")
            print(f"  資料筆數: {len(data):,}")
            print(f"  日期範圍: {data['Date'].min().date()} 到 {data['Date'].max().date()}")
            print(f"  最早開盤價: ${data.iloc[0]['Open']:.2f}")
            print(f"  最早收盤價: ${data.iloc[0]['Close']:.2f}")
            print(f"  最新開盤價: ${data.iloc[-1]['Open']:.2f}")
            print(f"  最新收盤價: ${data.iloc[-1]['Close']:.2f}")

            # Show example of early prices to verify they're actual prices
            print(f"\n📋 前5天實際價格範例:")
            print(data[['Date', 'Open', 'High', 'Low', 'Close']].head().to_string(index=False))

            return data

        except Exception as e:
            print(f"❌ 錯誤: {str(e)}")
            return None

    def download_multiple_etfs(self, tickers):
        """Download actual prices for multiple ETFs"""
        results = {}

        print("\n" + "="*70)
        print("批次下載 ETF 實際歷史價格")
        print("="*70)

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] 處理 {ticker}")

            if ticker in self.ETF_INFO:
                info = self.ETF_INFO[ticker]
                print(f"ETF 名稱: {info['name']}")
                print(f"說明: {info['description']}")
                print(f"首次交易日: {info['inception_date']}")

            data = self.download_actual_prices(ticker)
            results[ticker] = {
                'success': data is not None,
                'data': data
            }

        # Summary report
        print("\n" + "="*70)
        print("下載總結報告")
        print("="*70)

        for ticker, result in results.items():
            if result['success']:
                data = result['data']
                print(f"\n✅ {ticker}:")
                print(f"   資料筆數: {len(data):,}")
                print(f"   最早日期: {data['Date'].min().date()}")
                print(f"   最新日期: {data['Date'].max().date()}")
                print(f"   檔案位置: data/raw/{ticker}_actual.csv")

        print("\n" + "📝 重要說明 ".center(70, '='))
        print("已下載的是實際歷史價格（未經股息調整）")
        print("這些價格應該與您的券商軟體顯示的歷史價格相符")
        print("="*70)

        return results


def main():
    """Main function"""
    downloader = ETFActualPriceDownloader()

    # Download VOO, QQQ, and UPRO actual prices
    etfs_to_download = ['VOO', 'QQQ', 'UPRO']
    results = downloader.download_multiple_etfs(etfs_to_download)

    print("\n" + "✅ 所有實際價格下載任務完成！".center(70, '='))


if __name__ == "__main__":
    main()
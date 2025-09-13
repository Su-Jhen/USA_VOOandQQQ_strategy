"""
ETF Data Downloader Service
Downloads historical data from Yahoo Finance for VOO and QQQ ETFs
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


class ETFDataDownloader:
    """Service for downloading ETF historical data from Yahoo Finance"""

    def __init__(self, data_dir=None):
        """
        Initialize the data downloader

        Args:
            data_dir: Directory to save downloaded data.
                     Defaults to project's data/raw directory
        """
        if data_dir is None:
            # Get project root directory (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            data_dir = project_root / "data" / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data will be saved to: {self.data_dir}")

    def download_etf_data(self, ticker, start_date="2005-01-01", end_date=None):
        """
        Download historical data for a specific ETF

        Args:
            ticker: ETF ticker symbol (e.g., 'VOO', 'QQQ')
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data. Defaults to today

        Returns:
            DataFrame with historical data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n正在下載 {ticker} 的歷史資料...")
        print(f"期間: {start_date} 到 {end_date}")

        try:
            # Download data from Yahoo Finance
            etf = yf.Ticker(ticker)
            data = etf.history(start=start_date, end=end_date)

            if data.empty:
                print(f"警告: 無法下載 {ticker} 的資料")
                return None

            # Add ticker column
            data['Ticker'] = ticker

            # Reset index to make Date a column
            data.reset_index(inplace=True)

            print(f"成功下載 {len(data)} 筆資料")
            print(f"資料範圍: {data['Date'].min().date()} 到 {data['Date'].max().date()}")

            # Display basic statistics
            print(f"\n{ticker} 基本統計:")
            print(f"  開盤價範圍: ${data['Open'].min():.2f} - ${data['Open'].max():.2f}")
            print(f"  收盤價範圍: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"  平均成交量: {data['Volume'].mean():,.0f}")

            return data

        except Exception as e:
            print(f"錯誤: 下載 {ticker} 資料時發生錯誤: {str(e)}")
            return None

    def save_data(self, data, ticker, format='csv'):
        """
        Save downloaded data to file

        Args:
            data: DataFrame with historical data
            ticker: ETF ticker symbol
            format: File format ('csv' or 'excel')

        Returns:
            Path to saved file
        """
        if data is None or data.empty:
            print(f"錯誤: 無資料可儲存")
            return None

        timestamp = datetime.now().strftime("%Y%m%d")

        if format == 'csv':
            filename = f"{ticker}_historical_data_{timestamp}.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath, index=False)
            print(f"資料已儲存為 CSV: {filepath}")

        elif format == 'excel':
            filename = f"{ticker}_historical_data_{timestamp}.xlsx"
            filepath = self.data_dir / filename
            data.to_excel(filepath, index=False, sheet_name=ticker)
            print(f"資料已儲存為 Excel: {filepath}")
        else:
            raise ValueError(f"不支援的格式: {format}")

        # Also save as latest version for easy access
        latest_filename = f"{ticker}_latest.csv"
        latest_filepath = self.data_dir / latest_filename
        data.to_csv(latest_filepath, index=False)
        print(f"最新版本已儲存: {latest_filepath}")

        return filepath

    def download_and_save_voo(self, start_date="2005-01-01", end_date="2025-12-31"):
        """
        Download and save VOO historical data

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with VOO data
        """
        print("="*50)
        print("開始下載 VOO (Vanguard S&P 500 ETF) 資料")
        print("="*50)

        # Download VOO data
        voo_data = self.download_etf_data("VOO", start_date, end_date)

        if voo_data is not None:
            # Save in both CSV and Excel formats
            self.save_data(voo_data, "VOO", format='csv')
            # self.save_data(voo_data, "VOO", format='excel')

            print(f"\n✅ VOO 資料下載完成!")
            print(f"   總共 {len(voo_data)} 筆交易日資料")

            # Show recent data
            print(f"\n最近 5 筆資料:")
            print(voo_data.tail()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

        return voo_data

    def download_and_save_qqq(self, start_date="2005-01-01", end_date="2025-12-31"):
        """
        Download and save QQQ historical data

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with QQQ data
        """
        print("="*50)
        print("開始下載 QQQ (Invesco QQQ Trust) 資料")
        print("="*50)

        # Download QQQ data
        qqq_data = self.download_etf_data("QQQ", start_date, end_date)

        if qqq_data is not None:
            # Save in both CSV and Excel formats
            self.save_data(qqq_data, "QQQ", format='csv')
            # self.save_data(qqq_data, "QQQ", format='excel')

            print(f"\n✅ QQQ 資料下載完成!")
            print(f"   總共 {len(qqq_data)} 筆交易日資料")

            # Show recent data
            print(f"\n最近 5 筆資料:")
            print(qqq_data.tail()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

        return qqq_data


def main():
    """Main function to download VOO and QQQ data"""

    # Create downloader instance
    downloader = ETFDataDownloader()

    # Download VOO data from 2005 to 2025
    print("\n" + "="*60)
    print("ETF 歷史資料下載程式")
    print("="*60)

    voo_data = downloader.download_and_save_voo(
        start_date="2005-01-01",
        end_date="2025-12-31"
    )

    # Optionally download QQQ data as well
    # qqq_data = downloader.download_and_save_qqq(
    #     start_date="2005-01-01",
    #     end_date="2025-12-31"
    # )

    print("\n" + "="*60)
    print("所有下載任務完成!")
    print("="*60)


if __name__ == "__main__":
    main()
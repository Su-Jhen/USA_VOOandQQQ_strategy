"""
ETF Full Historical Data Downloader
Downloads complete historical data for VOO and QQQ ETFs
Handles inception dates and data availability
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


class ETFFullDownloader:
    """Downloads complete historical data for ETFs"""

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
        }
    }

    def __init__(self, data_dir=None):
        """Initialize downloader"""
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            data_dir = project_root / "data" / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def check_etf_availability(self, ticker):
        """Check ETF data availability and inception date"""
        print(f"\n{'='*60}")
        print(f"檢查 {ticker} 資料可用性")
        print(f"{'='*60}")

        # Get ETF info
        if ticker in self.ETF_INFO:
            info = self.ETF_INFO[ticker]
            print(f"ETF 名稱: {info['name']}")
            print(f"說明: {info['description']}")
            print(f"首次交易日: {info['inception_date']}")

        # Try to download earliest possible data
        try:
            etf = yf.Ticker(ticker)

            # Download from earliest possible date
            data = etf.history(period="max")

            if not data.empty:
                first_date = data.index.min()
                last_date = data.index.max()
                total_days = len(data)

                print(f"\n實際可用資料:")
                print(f"  最早日期: {first_date.date()}")
                print(f"  最新日期: {last_date.date()}")
                print(f"  總交易日數: {total_days:,}")
                print(f"  資料年數: {(last_date - first_date).days / 365.25:.1f} 年")

                # Price range
                print(f"\n價格範圍:")
                print(f"  歷史最低: ${data['Low'].min():.2f}")
                print(f"  歷史最高: ${data['High'].max():.2f}")
                print(f"  最新收盤: ${data['Close'].iloc[-1]:.2f}")

                return True, first_date.date(), last_date.date(), total_days
            else:
                print(f"❌ 無法取得 {ticker} 的資料")
                return False, None, None, 0

        except Exception as e:
            print(f"❌ 錯誤: {str(e)}")
            return False, None, None, 0

    def download_complete_history(self, ticker, save_format='csv'):
        """Download complete historical data for an ETF"""
        print(f"\n{'='*60}")
        print(f"下載 {ticker} 完整歷史資料")
        print(f"{'='*60}")

        try:
            # Create ticker object
            etf = yf.Ticker(ticker)

            # Download all available historical data
            print(f"正在下載 {ticker} 的所有可用資料...")
            data = etf.history(period="max")

            if data.empty:
                print(f"❌ 無法下載 {ticker} 的資料")
                return None

            # Add ticker column
            data['Ticker'] = ticker

            # Reset index to make Date a column
            data.reset_index(inplace=True)

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if save_format == 'csv':
                # Save timestamped version
                filename = f"{ticker}_complete_history_{timestamp}.csv"
                filepath = self.data_dir / filename
                data.to_csv(filepath, index=False)
                print(f"✅ 完整歷史資料已儲存: {filepath}")

                # Save latest version
                latest_filename = f"{ticker}_complete.csv"
                latest_filepath = self.data_dir / latest_filename
                data.to_csv(latest_filepath, index=False)
                print(f"✅ 最新版本已儲存: {latest_filepath}")

            # Display summary
            print(f"\n下載完成統計:")
            print(f"  資料筆數: {len(data):,}")
            print(f"  日期範圍: {data['Date'].min().date()} 到 {data['Date'].max().date()}")
            print(f"  最早開盤價: ${data.iloc[0]['Open']:.2f}")
            print(f"  最新收盤價: ${data.iloc[-1]['Close']:.2f}")
            print(f"  價格成長: {(data.iloc[-1]['Close'] / data.iloc[0]['Open'] - 1) * 100:.1f}%")

            return data

        except Exception as e:
            print(f"❌ 錯誤: {str(e)}")
            return None

    def download_all_etfs(self):
        """Download complete history for VOO and QQQ"""
        results = {}

        print("\n" + "="*70)
        print("ETF 完整歷史資料下載程式")
        print("="*70)

        # Check and download VOO
        print("\n[1/2] 處理 VOO (Vanguard S&P 500 ETF)")
        voo_available, voo_start, voo_end, voo_count = self.check_etf_availability('VOO')
        if voo_available:
            voo_data = self.download_complete_history('VOO')
            results['VOO'] = {
                'success': voo_data is not None,
                'start_date': voo_start,
                'end_date': voo_end,
                'count': voo_count
            }

        # Check and download QQQ
        print("\n[2/2] 處理 QQQ (Invesco QQQ Trust)")
        qqq_available, qqq_start, qqq_end, qqq_count = self.check_etf_availability('QQQ')
        if qqq_available:
            qqq_data = self.download_complete_history('QQQ')
            results['QQQ'] = {
                'success': qqq_data is not None,
                'start_date': qqq_start,
                'end_date': qqq_end,
                'count': qqq_count
            }

        # Summary report
        print("\n" + "="*70)
        print("下載總結報告")
        print("="*70)

        for ticker, result in results.items():
            if result['success']:
                print(f"\n✅ {ticker}:")
                print(f"   日期範圍: {result['start_date']} 到 {result['end_date']}")
                print(f"   資料筆數: {result['count']:,}")
                print(f"   檔案位置: data/raw/{ticker}_complete.csv")

        # Note about VOO inception
        print("\n" + "📝 重要說明 ".center(70, '='))
        print("VOO ETF 於 2010 年 9 月 7 日開始交易，因此沒有 2005-2010 年的資料。")
        print("如需更早的 S&P 500 資料，可考慮使用 SPY ETF (1993 年開始交易)。")
        print("QQQ ETF 於 1999 年 3 月 10 日開始交易，有完整的 2005-2025 資料。")
        print("="*70)

        return results

    def download_spy_as_alternative(self):
        """Download SPY as an alternative for earlier S&P 500 data"""
        print("\n" + "="*70)
        print("下載 SPY ETF 作為 VOO 的歷史資料替代")
        print("SPY 是最早的 S&P 500 ETF，從 1993 年開始交易")
        print("="*70)

        spy_available, spy_start, spy_end, spy_count = self.check_etf_availability('SPY')
        if spy_available:
            spy_data = self.download_complete_history('SPY')
            if spy_data is not None:
                print(f"\n✅ SPY 資料下載完成")
                print(f"   可用於分析 2005-2010 年的 S&P 500 表現")
                return spy_data
        return None


def main():
    """Main function"""
    downloader = ETFFullDownloader()

    # Download all ETFs
    results = downloader.download_all_etfs()

    # Ask if user wants SPY as alternative
    print("\n" + "="*70)
    print("💡 建議：由於 VOO 從 2010 年才開始，")
    print("   您可以下載 SPY ETF 來獲得 2005-2010 的 S&P 500 資料")
    print("   SPY 和 VOO 都追蹤相同的 S&P 500 指數")
    print("="*70)

    # Automatically download SPY as well
    spy_data = downloader.download_spy_as_alternative()

    print("\n" + "✅ 所有下載任務完成！".center(70, '='))


if __name__ == "__main__":
    main()
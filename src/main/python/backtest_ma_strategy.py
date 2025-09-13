"""
MA交叉策略回測主程式
執行VOO/QQQ動態配置策略的完整回測
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import warnings
import sys
from pathlib import Path

# 添加模組路徑
sys.path.insert(0, str(Path(__file__).parent))

from core.backtest_engine import BacktestEngine, Portfolio
from strategies.ma_crossover import MAcrossoverStrategy, MAStrategy, create_ma_strategy_variants

warnings.filterwarnings('ignore')

# 設置繪圖風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MAStrategyBacktest(BacktestEngine):
    """MA策略回測器"""

    def __init__(self, strategy_params: MAStrategy = None, **kwargs):
        """
        初始化MA策略回測

        Args:
            strategy_params: MA策略參數
            **kwargs: 回測引擎參數
        """
        super().__init__(**kwargs)
        self.strategy = MAcrossoverStrategy(strategy_params)
        self.equity_curve = pd.DataFrame()
        self.benchmark_curve = pd.DataFrame()

    def run_backtest(self) -> Dict:
        """執行回測"""
        print("\n" + "="*60)
        print("開始執行MA交叉策略回測")
        print("="*60)

        # 載入數據
        print("\n📊 載入數據...")
        self.load_data(['VOO', 'QQQ'])

        # 生成交易信號
        print("\n📈 生成交易信號...")
        signals = self.strategy.generate_signals(
            self.data['VOO'],
            self.data['QQQ']
        )

        # 獲取年度加碼日期
        contribution_dates = self.get_annual_contribution_dates()
        print(f"\n💰 年度加碼日期: {len(contribution_dates)}個")

        # 初始化權益曲線記錄
        equity_records = []
        trades_log = []

        # 回測主循環
        print("\n🔄 執行交易...")
        last_rebalance_date = None
        min_rebalance_days = 5  # 最少間隔5個交易日

        for date, signal_row in signals.iterrows():
            # 跳過指標未準備好的日期
            if pd.isna(signal_row['MA_fast']) or pd.isna(signal_row['MA_slow']):
                continue

            # 更新價格
            current_prices = {
                'VOO': signal_row['VOO_Close'],
                'QQQ': signal_row['QQQ_Close']
            }
            self.portfolio.update_prices(current_prices)

            # 處理年度加碼
            if date in contribution_dates:
                self.portfolio.cash += self.annual_contribution
                print(f"  💵 {date.date()}: 年度加碼 ${self.annual_contribution:,.0f}")

            # 獲取當前權重
            current_weights = self.portfolio.get_position_weights()

            # 目標權重
            target_weights = {
                'VOO': signal_row['VOO_Target_Weight'],
                'QQQ': signal_row['QQQ_Target_Weight']
            }

            # 判斷是否需要再平衡
            need_rebalance = False

            # 1. 信號變化時再平衡
            if signal_row['Signal_Change']:
                need_rebalance = True

            # 2. 權重偏離過大時再平衡
            elif self.strategy.should_rebalance(current_weights, target_weights):
                # 檢查距離上次再平衡的時間
                if last_rebalance_date is None or (date - last_rebalance_date).days >= min_rebalance_days:
                    need_rebalance = True

            # 3. 年度加碼時再平衡
            if date in contribution_dates:
                need_rebalance = True

            # 執行再平衡
            if need_rebalance:
                # 計算目標持倉
                total_value = self.portfolio.total_value
                target_positions = self.strategy.calculate_position_sizes(
                    total_value, target_weights, current_prices
                )

                # 執行交易
                for symbol in ['VOO', 'QQQ']:
                    current_shares = self.portfolio.positions.get(symbol)
                    current_shares = current_shares.shares if current_shares else 0
                    target_shares = target_positions.get(symbol, 0)
                    shares_diff = target_shares - current_shares

                    if abs(shares_diff) > 0:
                        if shares_diff > 0:
                            # 買入
                            success = self.portfolio.buy(symbol, shares_diff, current_prices[symbol], date)
                            if success:
                                trades_log.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares_diff,
                                    'price': current_prices[symbol],
                                    'market_state': signal_row['Market_State']
                                })
                        else:
                            # 賣出
                            success = self.portfolio.sell(symbol, abs(shares_diff), current_prices[symbol], date)
                            if success:
                                trades_log.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'shares': abs(shares_diff),
                                    'price': current_prices[symbol],
                                    'market_state': signal_row['Market_State']
                                })

                last_rebalance_date = date

            # 記錄每日權益
            weights = self.portfolio.get_position_weights()
            equity_records.append({
                'date': date,
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'voo_weight': weights.get('VOO', 0),
                'qqq_weight': weights.get('QQQ', 0),
                'market_state': signal_row['Market_State'],
                'returns': self.portfolio.returns
            })

        # 建立權益曲線
        self.equity_curve = pd.DataFrame(equity_records)
        self.equity_curve.set_index('date', inplace=True)

        # 計算績效指標
        self.results = self.calculate_metrics()

        # 分析信號
        signal_stats = self.strategy.analyze_signals()

        # 交易統計
        trades_df = pd.DataFrame(trades_log)

        print(f"\n✅ 回測完成！")
        print(f"   最終資產價值: ${self.portfolio.total_value:,.2f}")
        print(f"   總報酬率: {self.portfolio.returns:.2f}%")
        print(f"   總交易次數: {len(trades_df)}")

        return {
            'metrics': self.results,
            'equity_curve': self.equity_curve,
            'trades': trades_df,
            'signal_stats': signal_stats
        }

    def run_buy_hold_benchmark(self, symbol: str = 'VOO') -> pd.DataFrame:
        """執行Buy & Hold基準策略"""
        print(f"\n執行 {symbol} Buy & Hold 基準策略...")

        # 初始化基準投資組合
        benchmark = Portfolio(self.initial_capital, self.commission)
        benchmark_records = []

        # 獲取數據
        data = self.data[symbol]
        contribution_dates = self.get_annual_contribution_dates()

        for date, row in data.iterrows():
            # 跳過開始日期之前
            if date < self.start_date:
                continue

            # 更新價格
            benchmark.update_prices({symbol: row['Close']})

            # 年度加碼
            if date in contribution_dates:
                benchmark.cash += self.annual_contribution

            # 首次或加碼後買入
            if benchmark.cash > self.commission:
                shares_to_buy = int((benchmark.cash - self.commission) / row['Close'])
                if shares_to_buy > 0:
                    benchmark.buy(symbol, shares_to_buy, row['Close'], date)

            # 記錄
            benchmark_records.append({
                'date': date,
                'total_value': benchmark.total_value,
                'returns': benchmark.returns
            })

        benchmark_curve = pd.DataFrame(benchmark_records)
        benchmark_curve.set_index('date', inplace=True)

        # 計算基準績效
        total_return = (benchmark_curve['total_value'].iloc[-1] /
                       benchmark_curve['total_value'].iloc[0] - 1) * 100
        years = (benchmark_curve.index[-1] - benchmark_curve.index[0]).days / 365.25
        cagr = (np.power(benchmark_curve['total_value'].iloc[-1] /
                        benchmark_curve['total_value'].iloc[0], 1/years) - 1) * 100

        print(f"   {symbol} Buy & Hold 總報酬: {total_return:.2f}%")
        print(f"   {symbol} Buy & Hold CAGR: {cagr:.2f}%")

        return benchmark_curve

    def plot_results(self, save_path: str = None):
        """繪製回測結果圖表"""
        if self.equity_curve.empty:
            print("無數據可繪製")
            return

        # 獲取基準曲線
        voo_benchmark = self.run_buy_hold_benchmark('VOO')
        qqq_benchmark = self.run_buy_hold_benchmark('QQQ')

        # 創建子圖
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('MA交叉策略回測結果', fontsize=16, fontweight='bold')

        # 1. 權益曲線比較
        ax1 = axes[0, 0]
        ax1.plot(self.equity_curve.index, self.equity_curve['total_value'],
                label='MA策略', linewidth=2, color='blue')
        ax1.plot(voo_benchmark.index, voo_benchmark['total_value'],
                label='VOO Buy&Hold', linewidth=1.5, alpha=0.7, color='green')
        ax1.plot(qqq_benchmark.index, qqq_benchmark['total_value'],
                label='QQQ Buy&Hold', linewidth=1.5, alpha=0.7, color='red')
        ax1.set_title('權益曲線比較')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('資產價值 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 報酬率比較
        ax2 = axes[0, 1]
        strategy_returns = self.equity_curve['returns']
        voo_returns = (voo_benchmark['total_value'] / voo_benchmark['total_value'].iloc[0] - 1) * 100
        qqq_returns = (qqq_benchmark['total_value'] / qqq_benchmark['total_value'].iloc[0] - 1) * 100

        ax2.plot(self.equity_curve.index, strategy_returns,
                label='MA策略', linewidth=2, color='blue')
        ax2.plot(voo_benchmark.index, voo_returns,
                label='VOO Buy&Hold', linewidth=1.5, alpha=0.7, color='green')
        ax2.plot(qqq_benchmark.index, qqq_returns,
                label='QQQ Buy&Hold', linewidth=1.5, alpha=0.7, color='red')
        ax2.set_title('累積報酬率')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('報酬率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 資產配置變化
        ax3 = axes[1, 0]
        ax3.fill_between(self.equity_curve.index,
                        0, self.equity_curve['voo_weight'] * 100,
                        label='VOO', alpha=0.6, color='green')
        ax3.fill_between(self.equity_curve.index,
                        self.equity_curve['voo_weight'] * 100,
                        (self.equity_curve['voo_weight'] + self.equity_curve['qqq_weight']) * 100,
                        label='QQQ', alpha=0.6, color='red')
        ax3.set_title('資產配置變化')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('配置比例 (%)')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 回撤分析
        ax4 = axes[1, 1]
        # 計算回撤
        cummax = self.equity_curve['total_value'].expanding().max()
        drawdown = (self.equity_curve['total_value'] - cummax) / cummax * 100
        ax4.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax4.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax4.set_title('策略回撤')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('回撤 (%)')
        ax4.grid(True, alpha=0.3)

        # 5. 市場狀態分布
        ax5 = axes[2, 0]
        state_counts = self.equity_curve['market_state'].value_counts()
        colors = {'BULL': 'green', 'BEAR': 'red', 'NEUTRAL': 'gray'}
        ax5.pie(state_counts.values, labels=state_counts.index,
               autopct='%1.1f%%', colors=[colors.get(x, 'blue') for x in state_counts.index])
        ax5.set_title('市場狀態分布')

        # 6. 月度報酬分布
        ax6 = axes[2, 1]
        monthly_returns = self.equity_curve['total_value'].resample('M').last().pct_change() * 100
        ax6.hist(monthly_returns.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax6.axvline(monthly_returns.mean(), color='red', linestyle='--',
                   label=f'平均: {monthly_returns.mean():.2f}%')
        ax6.set_title('月度報酬分布')
        ax6.set_xlabel('月報酬率 (%)')
        ax6.set_ylabel('頻率')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # 儲存圖表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 圖表已儲存: {save_path}")

        plt.show()

    def generate_detailed_report(self) -> str:
        """生成詳細回測報告"""
        report = self.generate_report()

        # 添加策略信息
        report += "\n" + self.strategy.get_strategy_info()

        # 添加信號統計
        signal_stats = self.strategy.analyze_signals()
        if signal_stats:
            report += f"""
📊 信號統計
- 牛市天數: {signal_stats.get('bull_days', 0)}
- 熊市天數: {signal_stats.get('bear_days', 0)}
- 中性天數: {signal_stats.get('neutral_days', 0)}
- 信號變化次數: {signal_stats.get('signal_changes', 0)}
- 平均狀態持續天數: {signal_stats.get('avg_state_duration', 0):.1f}
"""

        return report


def run_multiple_strategies():
    """執行多個策略變體的回測"""
    print("\n" + "="*70)
    print("執行MA策略多參數回測")
    print("="*70)

    # 獲取策略變體
    variants = create_ma_strategy_variants()
    results_summary = []

    for name, params in variants.items():
        print(f"\n{'='*40}")
        print(f"測試策略: {name}")
        print(f"{'='*40}")

        # 執行回測
        backtest = MAStrategyBacktest(
            strategy_params=params,
            start_date='2010-09-09',
            end_date='2025-09-12'
        )

        results = backtest.run_backtest()
        metrics = results['metrics']

        # 記錄結果
        results_summary.append({
            'strategy': name,
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'volatility': metrics['annual_volatility'],
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'calmar': metrics['calmar_ratio'],
            'trades': metrics['num_trades']
        })

        # 打印簡要結果
        print(f"CAGR: {metrics['cagr']:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2f}%")

    # 創建結果比較表
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('sharpe', ascending=False)

    print("\n" + "="*70)
    print("策略比較結果")
    print("="*70)
    print(results_df.to_string())

    return results_df


def main():
    """主函數"""
    print("\n" + "🚀 VOO/QQQ MA交叉策略回測系統 🚀".center(70, '='))

    # 執行標準策略回測
    print("\n1. 執行標準MA策略回測")
    standard_params = MAStrategy(
        fast_period=50,
        slow_period=200,
        qqq_weight_bear=0.5,
        use_slope_confirm=False
    )

    backtest = MAStrategyBacktest(
        strategy_params=standard_params,
        start_date='2010-09-09',
        end_date='2025-09-12',
        initial_capital=10000,
        annual_contribution=3000
    )

    # 執行回測
    results = backtest.run_backtest()

    # 生成報告
    report = backtest.generate_detailed_report()
    print(report)

    # 繪製圖表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"output/ma_strategy_results_{timestamp}.png"
    backtest.plot_results(chart_path)

    # 執行多策略比較
    print("\n2. 執行多策略參數比較")
    comparison_results = run_multiple_strategies()

    # 儲存結果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 儲存詳細結果
    results['equity_curve'].to_csv(f"output/equity_curve_{timestamp}.csv")
    results['trades'].to_csv(f"output/trades_log_{timestamp}.csv")
    comparison_results.to_csv(f"output/strategy_comparison_{timestamp}.csv")

    # 儲存報告
    with open(f"output/backtest_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 所有結果已儲存到 output/ 目錄")
    print("="*70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
增強版策略A綜合測試系統
解決以下問題：
1. 為每組參數生成詳細交易記錄
2. 增加Buy & Hold基準策略（純VOO、純QQQ）
3. 生成圖表化的表現比較
4. 完整的資產變化追蹤
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from backtest_ma_strategy import MAStrategyBacktest
from strategies.ma_crossover import create_ma_parameter_matrix
from utils.bilingual_reporter import bilingual_reporter
from utils.plot_config_english import setup_plotting, get_english_labels

warnings.filterwarnings('ignore')
setup_plotting()

class ComprehensiveStrategyTest:
    """綜合策略測試系統"""

    def __init__(self, start_date='2010-09-09', end_date='2025-09-12'):
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}
        self.equity_curves = {}
        self.trade_logs = {}

    def run_buy_hold_benchmarks(self):
        """執行Buy & Hold基準策略"""
        print("執行Buy & Hold基準策略...")

        benchmarks = {}

        # 純VOO策略
        print("  測試純VOO Buy & Hold...")
        voo_backtest = MAStrategyBacktest(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=10000,
            annual_contribution=3000
        )
        voo_results = voo_backtest.run_buy_hold_benchmark('VOO')
        benchmarks['純VOO_Buy_Hold'] = {
            'equity_curve': voo_results,
            'strategy_type': 'buy_hold',
            'symbol': 'VOO'
        }

        # 純QQQ策略
        print("  測試純QQQ Buy & Hold...")
        qqq_backtest = MAStrategyBacktest(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=10000,
            annual_contribution=3000
        )
        qqq_results = qqq_backtest.run_buy_hold_benchmark('QQQ')
        benchmarks['純QQQ_Buy_Hold'] = {
            'equity_curve': qqq_results,
            'strategy_type': 'buy_hold',
            'symbol': 'QQQ'
        }

        return benchmarks

    def run_ma_strategies_with_details(self):
        """執行MA策略並生成詳細記錄"""
        print("執行MA策略詳細測試...")

        strategies = create_ma_parameter_matrix()
        results_summary = []

        for i, (name, params) in enumerate(strategies.items(), 1):
            print(f"  [{i}/60] 測試: {name}")

            try:
                # 執行回測
                backtest = MAStrategyBacktest(
                    strategy_params=params,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=10000,
                    annual_contribution=3000
                )

                results = backtest.run_backtest(show_progress=False)
                metrics = results['metrics']

                # 儲存詳細資料
                self.results[name] = {
                    'metrics': metrics,
                    'parameters': params.__dict__,
                    'strategy_type': 'ma_crossover'
                }

                self.equity_curves[name] = results['equity_curve']
                self.trade_logs[name] = results['trades']

                # 摘要記錄
                results_summary.append({
                    'strategy_name': name,
                    'fast_period': params.fast_period,
                    'slow_period': params.slow_period,
                    'qqq_weight_bear': params.qqq_weight_bear,
                    'use_slope_confirm': params.use_slope_confirm,
                    'total_return': metrics['total_return'],
                    'cagr': metrics['cagr'],
                    'volatility': metrics['annual_volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'num_trades': metrics['num_trades'],
                    'final_value': metrics['final_value']
                })

            except Exception as e:
                print(f"    錯誤: {e}")
                continue

        return pd.DataFrame(results_summary)

    def calculate_benchmark_metrics(self, benchmarks):
        """計算基準策略指標"""
        benchmark_summary = []

        for name, data in benchmarks.items():
            equity_curve = data['equity_curve']

            # 計算績效指標
            total_return = (equity_curve['total_value'].iloc[-1] /
                          equity_curve['total_value'].iloc[0] - 1) * 100

            years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
            cagr = (np.power(equity_curve['total_value'].iloc[-1] /
                           equity_curve['total_value'].iloc[0], 1/years) - 1) * 100

            # 計算波動率
            returns = equity_curve['returns'].dropna()
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

            # 計算最大回撤
            cumulative = (1 + returns / 100).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()

            calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

            benchmark_summary.append({
                'strategy_name': name,
                'fast_period': 0,
                'slow_period': 0,
                'qqq_weight_bear': 1.0 if 'QQQ' in name else 0.0,
                'use_slope_confirm': False,
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'num_trades': 0,  # Buy & Hold 不交易
                'final_value': equity_curve['total_value'].iloc[-1]
            })

            # 儲存到results中以便後續使用
            self.results[name] = {
                'metrics': {
                    'total_return': total_return,
                    'cagr': cagr,
                    'annual_volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': calmar_ratio,
                    'num_trades': 0,
                    'final_value': equity_curve['total_value'].iloc[-1]
                },
                'strategy_type': 'buy_hold'
            }

            self.equity_curves[name] = equity_curve
            self.trade_logs[name] = pd.DataFrame()  # Buy & Hold無交易記錄

        return pd.DataFrame(benchmark_summary)

    def create_performance_charts(self, all_results, output_dir):
        """創建表現圖表"""
        print("生成表現圖表...")

        # 重新設置字型以確保正確顯示
        setup_plotting()

        # 設置圖表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Strategy A Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        # 1. 權益曲線比較（前10名 + 基準）
        ax1 = axes[0, 0]
        top_strategies = all_results.nlargest(10, 'sharpe_ratio')['strategy_name'].tolist()
        benchmark_names = ['純VOO_Buy_Hold', '純QQQ_Buy_Hold']

        for strategy_name in top_strategies + benchmark_names:
            if strategy_name in self.equity_curves:
                equity_curve = self.equity_curves[strategy_name]
                label = strategy_name[:20] + '...' if len(strategy_name) > 20 else strategy_name

                if strategy_name in benchmark_names:
                    ax1.plot(equity_curve.index, equity_curve['total_value'],
                           linewidth=3, alpha=0.8, label=label)
                else:
                    ax1.plot(equity_curve.index, equity_curve['total_value'],
                           alpha=0.7, label=label)

        ax1.set_title('Equity Curves: Top 10 Strategies vs Benchmarks')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 風險報酬散點圖
        ax2 = axes[0, 1]

        # MA策略
        ma_strategies = all_results[~all_results['strategy_name'].str.contains('純')]
        ax2.scatter(ma_strategies['volatility'], ma_strategies['cagr'],
                   alpha=0.6, s=50, c='blue', label='MA Strategies')

        # 基準策略
        benchmarks_df = all_results[all_results['strategy_name'].str.contains('純')]
        ax2.scatter(benchmarks_df['volatility'], benchmarks_df['cagr'],
                   s=100, c='red', marker='s', label='Buy & Hold')

        # 標記最佳策略
        best_strategy = all_results.loc[all_results['sharpe_ratio'].idxmax()]
        ax2.scatter(best_strategy['volatility'], best_strategy['cagr'],
                   s=200, c='gold', marker='*', label=f"Best: {best_strategy['strategy_name'][:15]}...")

        ax2.set_xlabel('Annual Volatility (%)')
        ax2.set_ylabel('CAGR (%)')
        ax2.set_title('Risk-Return Scatter Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 夏普比率排名（前15名）
        ax3 = axes[0, 2]
        top_15 = all_results.nlargest(15, 'sharpe_ratio')

        colors = ['gold' if name in benchmark_names else 'skyblue' for name in top_15['strategy_name']]
        bars = ax3.barh(range(len(top_15)), top_15['sharpe_ratio'], color=colors)

        ax3.set_yticks(range(len(top_15)))
        ax3.set_yticklabels([name[:15] + '...' if len(name) > 15 else name
                            for name in top_15['strategy_name']], fontsize=8)
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_title('Top 15 Strategies by Sharpe Ratio')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. 最大回撤比較
        ax4 = axes[1, 0]
        top_10_dd = all_results.nsmallest(10, 'max_drawdown')  # 最小回撤（最好）

        colors = ['red' if name in benchmark_names else 'lightcoral' for name in top_10_dd['strategy_name']]
        bars = ax4.barh(range(len(top_10_dd)), top_10_dd['max_drawdown'], color=colors)

        ax4.set_yticks(range(len(top_10_dd)))
        ax4.set_yticklabels([name[:15] + '...' if len(name) > 15 else name
                            for name in top_10_dd['strategy_name']], fontsize=8)
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_title('Top 10 Strategies by Max Drawdown (Lower is Better)')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. MA參數熱力圖
        ax5 = axes[1, 1]
        ma_only = all_results[~all_results['strategy_name'].str.contains('純')]

        # 創建參數組合的平均夏普比率矩陣
        pivot_data = ma_only.groupby(['fast_period', 'slow_period'])['sharpe_ratio'].mean().unstack()

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax5, cbar_kws={'label': 'Average Sharpe Ratio'})
        ax5.set_title('MA Parameters Heatmap (Average Sharpe Ratio)')
        ax5.set_xlabel('Slow Period')
        ax5.set_ylabel('Fast Period')

        # 6. QQQ權重效果
        ax6 = axes[1, 2]
        ma_only_grouped = ma_only.groupby('qqq_weight_bear').agg({
            'cagr': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        })

        ax6_twin = ax6.twinx()

        line1 = ax6.plot(ma_only_grouped.index * 100, ma_only_grouped['cagr'],
                        'o-', color='blue', label='Average CAGR')
        line2 = ax6.plot(ma_only_grouped.index * 100, ma_only_grouped['sharpe_ratio'],
                        's-', color='green', label='Average Sharpe')
        line3 = ax6_twin.plot(ma_only_grouped.index * 100, ma_only_grouped['max_drawdown'],
                             '^-', color='red', label='Average Max DD')

        ax6.set_xlabel('QQQ Weight in Bear Market (%)')
        ax6.set_ylabel('CAGR (%) / Sharpe Ratio', color='blue')
        ax6_twin.set_ylabel('Max Drawdown (%)', color='red')
        ax6.set_title('QQQ Weight Impact Analysis')
        ax6.grid(True, alpha=0.3)

        # 合併圖例
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        # 儲存圖表
        chart_file = output_dir / f"comprehensive_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  圖表已儲存: {chart_file}")
        return str(chart_file)

    def save_detailed_results(self, all_results, benchmarks):
        """儲存詳細結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/strategies/ma_crossover/comprehensive")
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. 儲存總體結果比較
        all_results_sorted = all_results.sort_values('sharpe_ratio', ascending=False)
        results_file = output_dir / f"comprehensive_results_{timestamp}.csv"
        all_results_sorted.to_csv(results_file, index=False)
        saved_files['comprehensive_results'] = str(results_file)

        # 2. 儲存雙語版本
        results_bilingual = bilingual_reporter.create_bilingual_csv_headers(all_results_sorted)
        results_bilingual = bilingual_reporter.translate_categorical_values(results_bilingual)
        bilingual_file = output_dir / f"comprehensive_results_bilingual_{timestamp}.csv"
        results_bilingual.to_csv(bilingual_file, index=False, encoding='utf-8-sig')
        saved_files['comprehensive_results_bilingual'] = str(bilingual_file)

        # 3. 為每個策略儲存詳細交易記錄
        trades_dir = output_dir / "detailed_trades"
        trades_dir.mkdir(exist_ok=True)

        for strategy_name, trades_df in self.trade_logs.items():
            if not trades_df.empty:
                # 儲存原版交易記錄
                trade_file = trades_dir / f"trades_{strategy_name}_{timestamp}.csv"
                trades_df.to_csv(trade_file, index=False)

                # 儲存雙語版本
                trades_bilingual = bilingual_reporter.create_bilingual_csv_headers(trades_df)
                trades_bilingual = bilingual_reporter.translate_categorical_values(trades_bilingual)
                trade_bilingual_file = trades_dir / f"trades_{strategy_name}_bilingual_{timestamp}.csv"
                trades_bilingual.to_csv(trade_bilingual_file, index=False, encoding='utf-8-sig')

        saved_files['detailed_trades_dir'] = str(trades_dir)

        # 4. 為每個策略儲存權益曲線
        equity_dir = output_dir / "equity_curves"
        equity_dir.mkdir(exist_ok=True)

        for strategy_name, equity_df in self.equity_curves.items():
            equity_file = equity_dir / f"equity_{strategy_name}_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=True)

            # 雙語版本
            equity_bilingual = bilingual_reporter.create_bilingual_csv_headers(equity_df)
            equity_bilingual = bilingual_reporter.translate_categorical_values(equity_bilingual)
            equity_bilingual_file = equity_dir / f"equity_{strategy_name}_bilingual_{timestamp}.csv"
            equity_bilingual.to_csv(equity_bilingual_file, index=True, encoding='utf-8-sig')

        saved_files['equity_curves_dir'] = str(equity_dir)

        # 5. 生成圖表
        chart_file = self.create_performance_charts(all_results_sorted, output_dir)
        saved_files['performance_charts'] = chart_file

        # 6. 生成綜合分析報告
        report_file = self.generate_comprehensive_report(all_results_sorted, output_dir, timestamp)
        saved_files['analysis_report'] = str(report_file)

        return saved_files

    def generate_comprehensive_report(self, all_results, output_dir, timestamp):
        """生成綜合分析報告"""
        report_file = output_dir / f"comprehensive_analysis_{timestamp}.txt"

        # 找出關鍵策略
        best_strategy = all_results.iloc[0]
        best_benchmark = all_results[all_results['strategy_name'].str.contains('純')].iloc[0]

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""
策略A綜合測試報告（包含基準策略）
================================================================================

測試概況:
• MA策略測試: 60組參數組合
• 基準策略: 純VOO Buy & Hold、純QQQ Buy & Hold
• 測試期間: {self.start_date} 至 {self.end_date}
• 初始資金: $10,000
• 年度加碼: $3,000

最佳MA策略:
• 策略名稱: {best_strategy['strategy_name']}
• 快線週期: {best_strategy['fast_period']} 日
• 慢線週期: {best_strategy['slow_period']} 日
• QQQ熊市權重: {best_strategy['qqq_weight_bear']*100:.0f}%
• 年化報酬率: {best_strategy['cagr']:.2f}%
• 夏普比率: {best_strategy['sharpe_ratio']:.3f}
• 最大回撤: {best_strategy['max_drawdown']:.2f}%
• 最終價值: ${best_strategy['final_value']:,.2f}

最佳基準策略:
• 策略名稱: {best_benchmark['strategy_name']}
• 年化報酬率: {best_benchmark['cagr']:.2f}%
• 夏普比率: {best_benchmark['sharpe_ratio']:.3f}
• 最大回撤: {best_benchmark['max_drawdown']:.2f}%
• 最終價值: ${best_benchmark['final_value']:,.2f}

核心發現:
""")

            # 分析MA策略 vs 基準策略
            ma_strategies = all_results[~all_results['strategy_name'].str.contains('純')]
            benchmarks_df = all_results[all_results['strategy_name'].str.contains('純')]

            f.write("1. 動態配置 vs Buy & Hold比較:\n")

            voo_benchmark = benchmarks_df[benchmarks_df['strategy_name'].str.contains('VOO')]
            qqq_benchmark = benchmarks_df[benchmarks_df['strategy_name'].str.contains('QQQ')]

            if not voo_benchmark.empty:
                voo_perf = voo_benchmark.iloc[0]
                better_than_voo = len(ma_strategies[ma_strategies['sharpe_ratio'] > voo_perf['sharpe_ratio']])
                f.write(f"   • {better_than_voo}/60 MA策略夏普比率優於純VOO ({voo_perf['sharpe_ratio']:.3f})\n")
                f.write(f"   • 純VOO: CAGR {voo_perf['cagr']:.2f}%, 回撤 {voo_perf['max_drawdown']:.2f}%\n")

            if not qqq_benchmark.empty:
                qqq_perf = qqq_benchmark.iloc[0]
                better_than_qqq = len(ma_strategies[ma_strategies['sharpe_ratio'] > qqq_perf['sharpe_ratio']])
                f.write(f"   • {better_than_qqq}/60 MA策略夏普比率優於純QQQ ({qqq_perf['sharpe_ratio']:.3f})\n")
                f.write(f"   • 純QQQ: CAGR {qqq_perf['cagr']:.2f}%, 回撤 {qqq_perf['max_drawdown']:.2f}%\n")

            f.write(f"\n2. MA參數分析:\n")
            ma_param_analysis = ma_strategies.groupby(['fast_period', 'slow_period']).agg({
                'cagr': 'mean',
                'sharpe_ratio': 'mean',
                'max_drawdown': 'mean'
            }).round(3)

            best_ma_combo = ma_param_analysis.sort_values('sharpe_ratio', ascending=False).head(3)
            f.write("   最佳MA參數組合（平均夏普比率）:\n")
            for (fast, slow), metrics in best_ma_combo.iterrows():
                f.write(f"   • {fast}/{slow}: CAGR {metrics['cagr']:.2f}%, 夏普 {metrics['sharpe_ratio']:.3f}\n")

            f.write(f"\n3. QQQ權重分析:\n")
            weight_analysis = ma_strategies.groupby('qqq_weight_bear').agg({
                'cagr': 'mean',
                'sharpe_ratio': 'mean',
                'max_drawdown': 'mean'
            }).round(3)

            best_weight = weight_analysis.sort_values('sharpe_ratio', ascending=False).head(3)
            f.write("   最佳QQQ權重（平均夏普比率）:\n")
            for weight, metrics in best_weight.iterrows():
                f.write(f"   • {weight*100:.0f}%: CAGR {metrics['cagr']:.2f}%, 夏普 {metrics['sharpe_ratio']:.3f}\n")

            f.write(f"\n報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return report_file

def main():
    """主函數"""
    print("=" * 80)
    print("策略A綜合測試系統（包含基準策略）")
    print("=" * 80)

    # 創建測試器
    tester = ComprehensiveStrategyTest()

    # 1. 執行基準策略
    print("\n第1步: 執行Buy & Hold基準策略")
    benchmarks = tester.run_buy_hold_benchmarks()
    benchmark_results = tester.calculate_benchmark_metrics(benchmarks)

    # 2. 執行MA策略
    print("\n第2步: 執行MA策略參數測試")
    ma_results = tester.run_ma_strategies_with_details()

    # 3. 合併所有結果
    all_results = pd.concat([benchmark_results, ma_results], ignore_index=True)
    all_results = all_results.sort_values('sharpe_ratio', ascending=False)

    # 4. 儲存詳細結果和生成圖表
    print("\n第3步: 生成圖表和儲存結果")
    saved_files = tester.save_detailed_results(all_results, benchmarks)

    # 5. 顯示結果摘要
    print("\n" + "=" * 80)
    print("測試完成！結果摘要:")
    print("=" * 80)

    print(f"\n最佳策略 (夏普比率):")
    best = all_results.iloc[0]
    print(f"  {best['strategy_name']}: 夏普 {best['sharpe_ratio']:.3f}, CAGR {best['cagr']:.2f}%")

    print(f"\n基準策略表現:")
    benchmarks_df = all_results[all_results['strategy_name'].str.contains('純')]
    for _, row in benchmarks_df.iterrows():
        print(f"  {row['strategy_name']}: 夏普 {row['sharpe_ratio']:.3f}, CAGR {row['cagr']:.2f}%")

    print(f"\n檔案儲存位置:")
    for file_type, path in saved_files.items():
        print(f"  {file_type}: {path}")

    print("\n" + "=" * 80)
    print("✅ 綜合測試完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
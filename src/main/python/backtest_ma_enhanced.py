"""
增強版MA策略回測系統
詳細記錄所有參數組合的測試結果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import warnings
import sys
import json
from pathlib import Path

# 添加模組路徑
sys.path.insert(0, str(Path(__file__).parent))

from core.backtest_engine import BacktestEngine, Portfolio
from strategies.ma_crossover import MAcrossoverStrategy, MAStrategy, create_ma_strategy_variants
from backtest_ma_strategy import MAStrategyBacktest

warnings.filterwarnings('ignore')

# 設置繪圖風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnhancedBacktestReporter:
    """增強版回測報告生成器"""

    def __init__(self):
        self.all_results = {}
        self.detailed_reports = {}
        self.parameter_grid = []
        self.comparison_metrics = pd.DataFrame()

    def run_parameter_grid_backtest(self,
                                   fast_periods: List[int] = [20, 50, 100],
                                   slow_periods: List[int] = [100, 150, 200],
                                   qqq_weights_bear: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
                                   use_slope_confirms: List[bool] = [True, False]):
        """
        執行參數網格回測

        Args:
            fast_periods: 快線週期列表
            slow_periods: 慢線週期列表
            qqq_weights_bear: 熊市QQQ權重列表
            use_slope_confirms: 是否使用斜率確認列表
        """
        print("\n" + "="*80)
        print("執行完整參數網格回測")
        print("="*80)

        # 生成參數組合
        param_combinations = []
        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow:  # 確保快線週期小於慢線
                    for qqq_weight in qqq_weights_bear:
                        for use_slope in use_slope_confirms:
                            param_combinations.append({
                                'fast_period': fast,
                                'slow_period': slow,
                                'qqq_weight_bear': qqq_weight,
                                'use_slope_confirm': use_slope
                            })

        total_combinations = len(param_combinations)
        print(f"\n總共測試 {total_combinations} 種參數組合")
        print("="*80)

        # 執行每個參數組合的回測
        for idx, params in enumerate(param_combinations, 1):
            param_name = f"MA_{params['fast_period']}_{params['slow_period']}_QQQ{int(params['qqq_weight_bear']*100)}_{'Slope' if params['use_slope_confirm'] else 'NoSlope'}"

            print(f"\n[{idx}/{total_combinations}] 測試參數組合: {param_name}")
            print(f"  快線: {params['fast_period']}日")
            print(f"  慢線: {params['slow_period']}日")
            print(f"  熊市QQQ: {params['qqq_weight_bear']*100:.0f}%")
            print(f"  斜率確認: {'是' if params['use_slope_confirm'] else '否'}")

            # 創建策略參數
            strategy_params = MAStrategy(
                fast_period=params['fast_period'],
                slow_period=params['slow_period'],
                qqq_weight_bear=params['qqq_weight_bear'],
                use_slope_confirm=params['use_slope_confirm']
            )

            # 執行回測
            try:
                backtest = MAStrategyBacktest(
                    strategy_params=strategy_params,
                    start_date='2010-09-09',
                    end_date='2025-09-12',
                    initial_capital=10000,
                    annual_contribution=3000
                )

                results = backtest.run_backtest()

                # 儲存詳細結果
                self.all_results[param_name] = {
                    'parameters': params,
                    'metrics': results['metrics'],
                    'equity_curve': results['equity_curve'],
                    'trades': results['trades'],
                    'signal_stats': results['signal_stats'],
                    'backtest': backtest
                }

                # 生成詳細報告
                detailed_report = backtest.generate_detailed_report()
                self.detailed_reports[param_name] = detailed_report

                # 顯示簡要結果
                metrics = results['metrics']
                print(f"  ✅ CAGR: {metrics['cagr']:.2f}%")
                print(f"  ✅ 夏普: {metrics['sharpe_ratio']:.3f}")
                print(f"  ✅ 最大回撤: {metrics['max_drawdown']:.2f}%")

            except Exception as e:
                print(f"  ❌ 回測失敗: {str(e)}")
                continue

        # 生成比較報告
        self._generate_comparison_report()

    def _generate_comparison_report(self):
        """生成綜合比較報告"""
        comparison_data = []

        for name, result in self.all_results.items():
            params = result['parameters']
            metrics = result['metrics']
            signal_stats = result['signal_stats']

            comparison_data.append({
                'strategy_name': name,
                'fast_period': params['fast_period'],
                'slow_period': params['slow_period'],
                'qqq_weight_bear': params['qqq_weight_bear'],
                'use_slope_confirm': params['use_slope_confirm'],
                'total_return': metrics['total_return'],
                'cagr': metrics['cagr'],
                'volatility': metrics['annual_volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'calmar_ratio': metrics['calmar_ratio'],
                'num_trades': metrics['num_trades'],
                'final_value': metrics['final_value'],
                'bull_days': signal_stats.get('bull_days', 0),
                'bear_days': signal_stats.get('bear_days', 0),
                'signal_changes': signal_stats.get('signal_changes', 0)
            })

        self.comparison_metrics = pd.DataFrame(comparison_data)
        self.comparison_metrics = self.comparison_metrics.sort_values('sharpe_ratio', ascending=False)

    def save_all_reports(self, output_dir: str = 'output'):
        """儲存所有報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. 儲存綜合比較CSV
        comparison_file = output_path / f"all_strategies_comparison_{timestamp}.csv"
        self.comparison_metrics.to_csv(comparison_file, index=False)
        print(f"\n✅ 綜合比較表已儲存: {comparison_file}")

        # 2. 儲存詳細報告（所有策略）
        detailed_report_file = output_path / f"all_strategies_detailed_report_{timestamp}.txt"
        with open(detailed_report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("完整參數組合回測詳細報告\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # 寫入摘要
            f.write("📊 回測摘要\n")
            f.write(f"測試參數組合數: {len(self.all_results)}\n")
            f.write(f"最佳夏普比率: {self.comparison_metrics['sharpe_ratio'].max():.3f}\n")
            f.write(f"最高CAGR: {self.comparison_metrics['cagr'].max():.2f}%\n")
            f.write(f"最小回撤: {self.comparison_metrics['max_drawdown'].max():.2f}%\n")
            f.write("\n" + "="*80 + "\n\n")

            # 寫入前10名策略
            f.write("🏆 前10名策略（按夏普比率排序）\n")
            f.write("-"*80 + "\n")
            top10 = self.comparison_metrics.head(10)
            for idx, row in top10.iterrows():
                f.write(f"\n排名 {idx+1}: {row['strategy_name']}\n")
                f.write(f"  參數: MA {row['fast_period']}/{row['slow_period']}, ")
                f.write(f"熊市QQQ {row['qqq_weight_bear']*100:.0f}%, ")
                f.write(f"斜率確認: {'是' if row['use_slope_confirm'] else '否'}\n")
                f.write(f"  績效: CAGR {row['cagr']:.2f}%, ")
                f.write(f"夏普 {row['sharpe_ratio']:.3f}, ")
                f.write(f"最大回撤 {row['max_drawdown']:.2f}%\n")
                f.write(f"  交易: {row['num_trades']}次, ")
                f.write(f"最終價值 ${row['final_value']:,.2f}\n")

            f.write("\n" + "="*80 + "\n\n")

            # 寫入每個策略的詳細報告
            f.write("📋 所有策略詳細報告\n")
            f.write("="*80 + "\n")

            for name in self.comparison_metrics['strategy_name']:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"策略: {name}\n")
                f.write(f"{'='*60}\n")
                f.write(self.detailed_reports[name])
                f.write("\n")

        print(f"✅ 詳細報告已儲存: {detailed_report_file}")

        # 3. 儲存每個策略的權益曲線
        equity_curves_dir = output_path / f"equity_curves_{timestamp}"
        equity_curves_dir.mkdir(exist_ok=True)

        for name, result in self.all_results.items():
            equity_file = equity_curves_dir / f"{name}_equity.csv"
            result['equity_curve'].to_csv(equity_file)

        print(f"✅ 權益曲線已儲存到: {equity_curves_dir}")

        # 4. 儲存JSON格式的參數和結果（方便程式讀取）
        json_file = output_path / f"all_strategies_results_{timestamp}.json"
        json_data = {}
        for name, result in self.all_results.items():
            json_data[name] = {
                'parameters': result['parameters'],
                'metrics': result['metrics'],
                'signal_stats': result['signal_stats']
            }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON結果已儲存: {json_file}")

        # 5. 生成參數熱力圖
        self._generate_heatmaps(output_path, timestamp)

    def _generate_heatmaps(self, output_path: Path, timestamp: str):
        """生成參數熱力圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MA策略參數優化熱力圖', fontsize=16, fontweight='bold')

        # 準備數據
        df = self.comparison_metrics

        # 1. CAGR熱力圖 (Fast vs Slow)
        pivot_cagr = df.pivot_table(
            values='cagr',
            index='slow_period',
            columns='fast_period',
            aggfunc='mean'
        )
        sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=axes[0, 0], cbar_kws={'label': 'CAGR (%)'})
        axes[0, 0].set_title('CAGR - MA週期組合')
        axes[0, 0].set_xlabel('快線週期')
        axes[0, 0].set_ylabel('慢線週期')

        # 2. 夏普比率熱力圖 (Fast vs QQQ Weight)
        pivot_sharpe = df.pivot_table(
            values='sharpe_ratio',
            index='qqq_weight_bear',
            columns='fast_period',
            aggfunc='mean'
        )
        sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn',
                   ax=axes[0, 1], cbar_kws={'label': '夏普比率'})
        axes[0, 1].set_title('夏普比率 - 快線週期 vs QQQ權重')
        axes[0, 1].set_xlabel('快線週期')
        axes[0, 1].set_ylabel('熊市QQQ權重')

        # 3. 最大回撤熱力圖 (Slow vs QQQ Weight)
        pivot_dd = df.pivot_table(
            values='max_drawdown',
            index='qqq_weight_bear',
            columns='slow_period',
            aggfunc='mean'
        )
        sns.heatmap(pivot_dd, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   ax=axes[1, 0], cbar_kws={'label': '最大回撤 (%)'})
        axes[1, 0].set_title('最大回撤 - 慢線週期 vs QQQ權重')
        axes[1, 0].set_xlabel('慢線週期')
        axes[1, 0].set_ylabel('熊市QQQ權重')

        # 4. 交易次數熱力圖
        pivot_trades = df.pivot_table(
            values='num_trades',
            index='slow_period',
            columns='fast_period',
            aggfunc='mean'
        )
        sns.heatmap(pivot_trades, annot=True, fmt='.0f', cmap='YlOrRd',
                   ax=axes[1, 1], cbar_kws={'label': '交易次數'})
        axes[1, 1].set_title('交易次數 - MA週期組合')
        axes[1, 1].set_xlabel('快線週期')
        axes[1, 1].set_ylabel('慢線週期')

        plt.tight_layout()
        heatmap_file = output_path / f"parameter_heatmaps_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 參數熱力圖已儲存: {heatmap_file}")

    def generate_performance_comparison_chart(self, output_path: str = 'output'):
        """生成績效比較圖表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 選擇前5名策略進行比較
        top5 = self.comparison_metrics.head(5)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('前5名策略績效比較', fontsize=14, fontweight='bold')

        # 1. CAGR比較
        ax1 = axes[0, 0]
        strategies = [name.split('_')[0] + '_' + name.split('_')[1] + '/' + name.split('_')[2]
                     for name in top5['strategy_name']]
        ax1.bar(range(len(strategies)), top5['cagr'].values)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_title('年化報酬率 (CAGR)')
        ax1.set_ylabel('CAGR (%)')
        ax1.grid(True, alpha=0.3)

        # 2. 風險調整報酬
        ax2 = axes[0, 1]
        ax2.scatter(top5['volatility'], top5['cagr'], s=100)
        for i, name in enumerate(top5['strategy_name']):
            ax2.annotate(f"{i+1}", (top5['volatility'].iloc[i], top5['cagr'].iloc[i]),
                        ha='center', va='center')
        ax2.set_xlabel('波動率 (%)')
        ax2.set_ylabel('CAGR (%)')
        ax2.set_title('風險-報酬散點圖')
        ax2.grid(True, alpha=0.3)

        # 3. 夏普比率比較
        ax3 = axes[1, 0]
        ax3.barh(range(len(strategies)), top5['sharpe_ratio'].values, color='green')
        ax3.set_yticks(range(len(strategies)))
        ax3.set_yticklabels(strategies)
        ax3.set_xlabel('夏普比率')
        ax3.set_title('夏普比率比較')
        ax3.grid(True, alpha=0.3)

        # 4. 回撤vs報酬
        ax4 = axes[1, 1]
        ax4.scatter(top5['max_drawdown'], top5['total_return'], s=100, color='red')
        for i, name in enumerate(top5['strategy_name']):
            ax4.annotate(f"{i+1}", (top5['max_drawdown'].iloc[i], top5['total_return'].iloc[i]),
                        ha='center', va='center')
        ax4.set_xlabel('最大回撤 (%)')
        ax4.set_ylabel('總報酬率 (%)')
        ax4.set_title('回撤 vs 報酬')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = Path(output_path) / f"top5_performance_comparison_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 績效比較圖已儲存: {chart_file}")


def run_comprehensive_backtest():
    """執行完整的參數組合回測"""
    print("\n" + "🚀 開始執行完整參數組合回測 🚀".center(80, '='))

    reporter = EnhancedBacktestReporter()

    # 定義要測試的參數範圍
    fast_periods = [20, 50, 100]
    slow_periods = [100, 150, 200, 252]
    qqq_weights_bear = [0.3, 0.4, 0.5, 0.6, 0.7]
    use_slope_confirms = [True, False]

    # 執行網格回測
    reporter.run_parameter_grid_backtest(
        fast_periods=fast_periods,
        slow_periods=slow_periods,
        qqq_weights_bear=qqq_weights_bear,
        use_slope_confirms=use_slope_confirms
    )

    # 儲存所有報告
    reporter.save_all_reports()

    # 生成績效比較圖
    reporter.generate_performance_comparison_chart()

    print("\n" + "="*80)
    print("✅ 完整參數組合回測完成！")
    print("="*80)

    # 顯示最佳策略
    print("\n🏆 最佳策略（按夏普比率）：")
    best = reporter.comparison_metrics.iloc[0]
    print(f"策略名稱: {best['strategy_name']}")
    print(f"參數: MA {best['fast_period']}/{best['slow_period']}, 熊市QQQ {best['qqq_weight_bear']*100:.0f}%")
    print(f"績效: CAGR {best['cagr']:.2f}%, 夏普 {best['sharpe_ratio']:.3f}, 最大回撤 {best['max_drawdown']:.2f}%")

    return reporter


def main():
    """主函數"""
    try:
        reporter = run_comprehensive_backtest()
        return reporter
    except KeyboardInterrupt:
        print("\n\n回測被使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
策略A增強版回測系統
實作完整的MA交叉策略參數矩陣測試和進階優化功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple

# 添加模組路徑
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backtest_ma_strategy import MAStrategyBacktest
from strategies.ma_crossover import (
    MAStrategy, MAcrossoverStrategy,
    create_ma_strategy_variants,
    create_ma_parameter_matrix,
    create_comprehensive_parameter_scan
)
from utils.bilingual_reporter import bilingual_reporter

warnings.filterwarnings('ignore')


class EnhancedMABacktest:
    """增強版MA策略回測器"""

    def __init__(self, start_date: str = '2010-09-09', end_date: str = '2025-09-12'):
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

    def run_parameter_matrix_test(self) -> pd.DataFrame:
        """
        執行完整的參數矩陣測試
        """
        print("\n" + "="*80)
        print(" 執行完整參數矩陣測試")
        print("="*80)

        # 獲取參數矩陣
        strategies = create_ma_parameter_matrix()
        print(f" 共 {len(strategies)} 組參數組合")

        results_list = []
        total_strategies = len(strategies)

        for i, (name, params) in enumerate(strategies.items(), 1):
            print(f"\n[{i}/{total_strategies}] 測試策略: {name}")
            print(f"   參數: {params.fast_period}/{params.slow_period}MA, QQQ權重={params.qqq_weight_bear*100:.0f}%")

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

                # 記錄結果
                result_row = {
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
                }

                results_list.append(result_row)

                # 顯示簡要結果
                print(f"    CAGR: {metrics['cagr']:.2f}%, 夏普: {metrics['sharpe_ratio']:.3f}, 回撤: {metrics['max_drawdown']:.2f}%")

                # 儲存詳細結果
                self.results[name] = {
                    'metrics': metrics,
                    'parameters': params.__dict__,
                    'equity_curve': results.get('equity_curve'),
                    'trades': results.get('trades')
                }

            except Exception as e:
                print(f"   ❌ 錯誤: {str(e)}")
                continue

        # 創建結果DataFrame
        results_df = pd.DataFrame(results_list)
        if not results_df.empty:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def run_advanced_optimization_test(self) -> pd.DataFrame:
        """
        測試進階優化功能
        """
        print("\n" + "="*80)
        print(" 執行進階優化功能測試")
        print("="*80)

        # 基礎策略參數（經典50/200）
        base_params = MAStrategy(
            fast_period=50,
            slow_period=200,
            qqq_weight_bear=0.5
        )

        # 測試不同的優化組合
        optimization_tests = [
            ("基礎版本", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5
            )),
            ("斜率確認", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5,
                use_slope_confirm=True
            )),
            ("交叉強度過濾", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5,
                use_crossover_filter=True, crossover_threshold=0.01
            )),
            ("持續時間確認", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5,
                use_duration_confirm=True, confirm_days=3
            )),
            ("動態權重", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5,
                use_dynamic_weight=True
            )),
            ("全部優化", MAStrategy(
                fast_period=50, slow_period=200, qqq_weight_bear=0.5,
                use_slope_confirm=True,
                use_crossover_filter=True, crossover_threshold=0.01,
                use_duration_confirm=True, confirm_days=3,
                use_dynamic_weight=True
            ))
        ]

        results_list = []

        for name, params in optimization_tests:
            print(f"\n 測試: {name}")

            try:
                backtest = MAStrategyBacktest(
                    strategy_params=params,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=10000,
                    annual_contribution=3000
                )

                results = backtest.run_backtest(show_progress=False)
                metrics = results['metrics']

                result_row = {
                    'optimization_type': name,
                    'total_return': metrics['total_return'],
                    'cagr': metrics['cagr'],
                    'volatility': metrics['annual_volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'num_trades': metrics['num_trades']
                }

                results_list.append(result_row)
                print(f"    CAGR: {metrics['cagr']:.2f}%, 夏普: {metrics['sharpe_ratio']:.3f}")

            except Exception as e:
                print(f"   ❌ 錯誤: {str(e)}")

        results_df = pd.DataFrame(results_list)
        if not results_df.empty:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def find_optimal_parameters(self, results_df: pd.DataFrame) -> Dict:
        """
        分析並找出最優參數組合
        """
        if results_df.empty:
            return {}

        print("\n" + "="*80)
        print(" 最優參數分析")
        print("="*80)

        # 按不同指標找最優
        best_by_metrics = {
            'highest_cagr': results_df.loc[results_df['cagr'].idxmax()],
            'highest_sharpe': results_df.loc[results_df['sharpe_ratio'].idxmax()],
            'lowest_drawdown': results_df.loc[results_df['max_drawdown'].idxmax()],  # 最大回撤最小
            'highest_calmar': results_df.loc[results_df['calmar_ratio'].idxmax()]
        }

        for metric, best_strategy in best_by_metrics.items():
            print(f"\n {metric.upper()}:")
            print(f"   策略: {best_strategy['strategy_name']}")
            print(f"   CAGR: {best_strategy['cagr']:.2f}%")
            print(f"   夏普比率: {best_strategy['sharpe_ratio']:.3f}")
            print(f"   最大回撤: {best_strategy['max_drawdown']:.2f}%")

        # 綜合評分（夏普比率權重70%，回撤控制30%）
        results_df['composite_score'] = (
            results_df['sharpe_ratio'] * 0.7 +
            (1 - abs(results_df['max_drawdown'])/100) * 0.3
        )

        best_overall = results_df.loc[results_df['composite_score'].idxmax()]
        print(f"\n 綜合最優策略:")
        print(f"   策略: {best_overall['strategy_name']}")
        print(f"   綜合評分: {best_overall['composite_score']:.3f}")
        print(f"   CAGR: {best_overall['cagr']:.2f}%")
        print(f"   夏普比率: {best_overall['sharpe_ratio']:.3f}")
        print(f"   最大回撤: {best_overall['max_drawdown']:.2f}%")

        return {
            'best_by_metrics': best_by_metrics,
            'best_overall': best_overall,
            'top_10': results_df.head(10)
        }

    def save_comprehensive_results(self,
                                 matrix_results: pd.DataFrame,
                                 optimization_results: pd.DataFrame,
                                 analysis: Dict) -> Dict[str, str]:
        """
        儲存所有結果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/strategies/ma_crossover/enhanced")
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. 參數矩陣結果
        if not matrix_results.empty:
            # 原版
            matrix_file = output_dir / f"parameter_matrix_results_{timestamp}.csv"
            matrix_results.to_csv(matrix_file, index=False)

            # 雙語版
            matrix_bilingual = bilingual_reporter.create_bilingual_csv_headers(matrix_results)
            matrix_bilingual = bilingual_reporter.translate_categorical_values(matrix_bilingual)
            matrix_bilingual_file = output_dir / f"parameter_matrix_bilingual_{timestamp}.csv"
            matrix_bilingual.to_csv(matrix_bilingual_file, index=False, encoding='utf-8-sig')

            saved_files['parameter_matrix'] = str(matrix_file)
            saved_files['parameter_matrix_bilingual'] = str(matrix_bilingual_file)

        # 2. 優化功能結果
        if not optimization_results.empty:
            opt_file = output_dir / f"optimization_results_{timestamp}.csv"
            optimization_results.to_csv(opt_file, index=False)
            saved_files['optimization_results'] = str(opt_file)

        # 3. 分析報告
        if analysis:
            report_file = output_dir / f"analysis_report_{timestamp}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_analysis_report(analysis, matrix_results, optimization_results))
            saved_files['analysis_report'] = str(report_file)

        return saved_files

    def _generate_analysis_report(self, analysis: Dict,
                                matrix_results: pd.DataFrame,
                                optimization_results: pd.DataFrame) -> str:
        """
        生成詳細分析報告
        """
        report = f"""
================================================================================
策略A（MA交叉法）完整測試報告 / Strategy A (MA Crossover) Comprehensive Test Report
================================================================================

測試概況 / Test Overview
────────────────────────────────────────────────────────────
• 參數矩陣測試策略數 / Parameter Matrix Tests: {len(matrix_results)} 組
• 優化功能測試數 / Optimization Tests: {len(optimization_results)} 組
• 測試期間 / Test Period: {self.start_date} 至 / to {self.end_date}
• 初始資金 / Initial Capital: $10,000
• 年度加碼 / Annual Contribution: $3,000

最優策略分析 / Optimal Strategy Analysis
────────────────────────────────────────────────────────────
"""

        if 'best_overall' in analysis:
            best = analysis['best_overall']
            report += f"""
綜合最優策略 / Best Overall Strategy:
• 策略名稱 / Strategy Name: {best['strategy_name']}
• 快線週期 / Fast MA: {best['fast_period']} 日 / days
• 慢線週期 / Slow MA: {best['slow_period']} 日 / days
• QQQ熊市權重 / Bear Market QQQ Weight: {best['qqq_weight_bear']*100:.0f}%
• 斜率確認 / Slope Confirmation: {'是 / Yes' if best['use_slope_confirm'] else '否 / No'}

績效指標 / Performance Metrics:
• 年化報酬率 / CAGR: {best['cagr']:.2f}%
• 夏普比率 / Sharpe Ratio: {best['sharpe_ratio']:.3f}
• 最大回撤 / Max Drawdown: {best['max_drawdown']:.2f}%
• Calmar比率 / Calmar Ratio: {best['calmar_ratio']:.3f}
• 交易次數 / Number of Trades: {best['num_trades']}
• 最終價值 / Final Value: ${best['final_value']:,.2f}

\"\"\"

        # 前10名策略
        if 'top_10' in analysis:
            report += """
前10名策略 / Top 10 Strategies
────────────────────────────────────────────────────────────
排名 / Rank | 策略名稱 / Strategy | CAGR | 夏普 / Sharpe | 回撤 / Drawdown
────────────────────────────────────────────────────────────────────────────────
"""
            for i, (_, row) in enumerate(analysis['top_10'].iterrows(), 1):
                report += f"{i:2d}     | {row['strategy_name']:<25} | {row['cagr']:5.2f}% | {row['sharpe_ratio']:6.3f} | {row['max_drawdown']:6.2f}%\n"

        report += """

參數分析 / Parameter Analysis
────────────────────────────────────────────────────────────
"""

        if not matrix_results.empty:
            # MA週期分析
            ma_analysis = matrix_results.groupby(['fast_period', 'slow_period']).agg({
                'cagr': 'mean',
                'sharpe_ratio': 'mean',
                'max_drawdown': 'mean'
            }).round(3)

            report += "\n最佳MA週期組合 / Best MA Period Combinations:\n"
            best_ma_combos = ma_analysis.sort_values('sharpe_ratio', ascending=False).head(5)
            for (fast, slow), metrics in best_ma_combos.iterrows():
                report += f"   {fast}/{slow}: CAGR {metrics['cagr']:.2f}%, 夏普 {metrics['sharpe_ratio']:.3f}\n"

            # QQQ權重分析
            weight_analysis = matrix_results.groupby('qqq_weight_bear').agg({
                'cagr': 'mean',
                'sharpe_ratio': 'mean',
                'max_drawdown': 'mean'
            }).round(3)

            report += "\n最佳QQQ權重 / Best QQQ Weight:\n"
            for weight, metrics in weight_analysis.iterrows():
                report += f"   {weight*100:.0f}%: CAGR {metrics['cagr']:.2f}%, 夏普 {metrics['sharpe_ratio']:.3f}\n"

        report += f"""

{'='*80}
報告生成時間 / Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        return report


def main():
    """主函數"""
    print("\n" + " 策略A增強版回測系統".center(80, '='))

    # 創建回測器
    enhanced_backtest = EnhancedMABacktest()

    # 1. 執行參數矩陣測試
    print("\n 階段1: 參數矩陣測試")
    matrix_results = enhanced_backtest.run_parameter_matrix_test()

    # 2. 執行進階優化測試
    print("\n 階段2: 進階優化測試")
    optimization_results = enhanced_backtest.run_advanced_optimization_test()

    # 3. 分析最優參數
    analysis = enhanced_backtest.find_optimal_parameters(matrix_results)

    # 4. 儲存結果
    saved_files = enhanced_backtest.save_comprehensive_results(
        matrix_results, optimization_results, analysis
    )

    print("\n" + "="*80)
    print(" 完整測試完成！")
    print(" 結果檔案:")
    for file_type, file_path in saved_files.items():
        print(f"   {file_type}: {file_path}")
    print("="*80)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
策略A完整參數測試執行器
避免字符編碼問題的簡化版本
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from backtest_ma_strategy import MAStrategyBacktest
from strategies.ma_crossover import create_ma_parameter_matrix
from utils.bilingual_reporter import bilingual_reporter

def run_full_strategy_a_test():
    """執行策略A的完整60組參數測試"""
    print("=" * 80)
    print("策略A完整參數矩陣測試")
    print("=" * 80)

    # 獲取所有參數組合
    strategies = create_ma_parameter_matrix()
    print(f"共 {len(strategies)} 組參數組合")

    results_list = []
    total_strategies = len(strategies)

    for i, (name, params) in enumerate(strategies.items(), 1):
        print(f"\n[{i}/{total_strategies}] 測試策略: {name}")
        print(f"   參數: {params.fast_period}/{params.slow_period}MA, QQQ權重={params.qqq_weight_bear*100:.0f}%")

        try:
            # 執行回測
            backtest = MAStrategyBacktest(
                strategy_params=params,
                start_date='2010-09-09',
                end_date='2025-09-12',
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

            # 顯示結果
            print(f"   CAGR: {metrics['cagr']:.2f}%, 夏普: {metrics['sharpe_ratio']:.3f}, 回撤: {metrics['max_drawdown']:.2f}%")

        except Exception as e:
            print(f"   錯誤: {str(e)}")
            continue

    # 創建結果DataFrame
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    return results_df

def analyze_results(results_df):
    """分析測試結果"""
    if results_df.empty:
        print("無結果可分析")
        return

    print("\n" + "=" * 80)
    print("結果分析")
    print("=" * 80)

    # 最佳策略
    best_sharpe = results_df.iloc[0]
    print(f"\n最佳夏普比率策略:")
    print(f"   策略: {best_sharpe['strategy_name']}")
    print(f"   CAGR: {best_sharpe['cagr']:.2f}%")
    print(f"   夏普比率: {best_sharpe['sharpe_ratio']:.3f}")
    print(f"   最大回撤: {best_sharpe['max_drawdown']:.2f}%")
    print(f"   交易次數: {best_sharpe['num_trades']}")

    # 前10名
    print(f"\n前10名策略:")
    print("排名 | 策略名稱                    | CAGR  | 夏普  | 回撤  | 交易次數")
    print("-" * 80)
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{i:2d}   | {row['strategy_name'][:25]:<25} | {row['cagr']:5.2f}% | {row['sharpe_ratio']:5.3f} | {row['max_drawdown']:5.2f}% | {row['num_trades']:4d}")

    return best_sharpe

def save_results(results_df, best_strategy):
    """儲存結果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/strategies/ma_crossover/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 儲存完整結果
    results_file = output_dir / f"full_parameter_test_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    # 儲存雙語版本
    results_bilingual = bilingual_reporter.create_bilingual_csv_headers(results_df)
    results_bilingual = bilingual_reporter.translate_categorical_values(results_bilingual)
    bilingual_file = output_dir / f"full_parameter_test_bilingual_{timestamp}.csv"
    results_bilingual.to_csv(bilingual_file, index=False, encoding='utf-8-sig')

    # 生成分析報告
    report_file = output_dir / f"strategy_a_analysis_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""
策略A（MA交叉法）完整測試報告
================================================================================

測試概況:
• 測試參數組合: {len(results_df)} 組
• 測試期間: 2010-09-09 至 2025-09-12
• 初始資金: $10,000
• 年度加碼: $3,000

最佳策略:
• 策略名稱: {best_strategy['strategy_name']}
• 快線週期: {best_strategy['fast_period']} 日
• 慢線週期: {best_strategy['slow_period']} 日
• QQQ熊市權重: {best_strategy['qqq_weight_bear']*100:.0f}%
• 斜率確認: {'是' if best_strategy['use_slope_confirm'] else '否'}

績效指標:
• 年化報酬率: {best_strategy['cagr']:.2f}%
• 夏普比率: {best_strategy['sharpe_ratio']:.3f}
• 最大回撤: {best_strategy['max_drawdown']:.2f}%
• Calmar比率: {best_strategy['calmar_ratio']:.3f}
• 總交易次數: {best_strategy['num_trades']}
• 最終價值: ${best_strategy['final_value']:,.2f}

參數分析:
""")

        # MA週期分析
        ma_analysis = results_df.groupby(['fast_period', 'slow_period']).agg({
            'cagr': 'mean',
            'sharpe_ratio': 'mean'
        }).round(3)

        f.write("\n最佳MA週期組合（按平均夏普比率）:\n")
        best_ma = ma_analysis.sort_values('sharpe_ratio', ascending=False).head(5)
        for (fast, slow), metrics in best_ma.iterrows():
            f.write(f"   {fast}/{slow}: 平均CAGR {metrics['cagr']:.2f}%, 平均夏普 {metrics['sharpe_ratio']:.3f}\n")

        # QQQ權重分析
        weight_analysis = results_df.groupby('qqq_weight_bear').agg({
            'cagr': 'mean',
            'sharpe_ratio': 'mean'
        }).round(3)

        f.write("\n最佳QQQ權重（按平均夏普比率）:\n")
        best_weight = weight_analysis.sort_values('sharpe_ratio', ascending=False)
        for weight, metrics in best_weight.iterrows():
            f.write(f"   {weight*100:.0f}%: 平均CAGR {metrics['cagr']:.2f}%, 平均夏普 {metrics['sharpe_ratio']:.3f}\n")

        f.write(f"\n報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n結果已儲存:")
    print(f"   完整結果: {results_file}")
    print(f"   雙語版本: {bilingual_file}")
    print(f"   分析報告: {report_file}")

def main():
    """主函數"""
    print("開始執行策略A完整參數測試...")

    # 執行測試
    results_df = run_full_strategy_a_test()

    if results_df.empty:
        print("測試失敗，無結果")
        return

    # 分析結果
    best_strategy = analyze_results(results_df)

    # 儲存結果
    save_results(results_df, best_strategy)

    print("\n" + "=" * 80)
    print("策略A完整測試完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
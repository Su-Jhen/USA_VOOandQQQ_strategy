#!/usr/bin/env python3
"""
執行完整參數組合回測
詳細記錄所有參數組合的測試結果
"""

import sys
from pathlib import Path

# 添加src/main/python到路徑
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from backtest_ma_enhanced import run_comprehensive_backtest

if __name__ == "__main__":
    print("="*80)
    print("MA策略完整參數組合回測系統")
    print("="*80)
    print("此程式將測試以下參數範圍:")
    print("- 快線週期: 20, 50, 100日")
    print("- 慢線週期: 100, 150, 200, 252日")
    print("- 熊市QQQ權重: 30%, 40%, 50%, 60%, 70%")
    print("- 斜率確認: 是/否")
    print(f"- 預計測試 {3*4*5*2} = 120 種參數組合")
    print("="*80)

    input("\n按 Enter 開始執行回測... (這可能需要幾分鐘時間)")

    try:
        reporter = run_comprehensive_backtest()

        print(f"\n🎉 成功完成 {len(reporter.all_results)} 個策略的回測！")
        print("\n生成的報告檔案:")
        print("- all_strategies_comparison_*.csv: 所有策略比較表")
        print("- all_strategies_detailed_report_*.txt: 詳細報告")
        print("- all_strategies_results_*.json: JSON格式結果")
        print("- parameter_heatmaps_*.png: 參數熱力圖")
        print("- top5_performance_comparison_*.png: 前5名績效比較")
        print("- equity_curves_*/: 每個策略的權益曲線")

    except KeyboardInterrupt:
        print("\n\n回測被使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
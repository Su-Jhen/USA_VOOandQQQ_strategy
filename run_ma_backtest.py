#!/usr/bin/env python3
"""
執行MA交叉策略回測
"""

import sys
from pathlib import Path

# 添加src/main/python到路徑
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from backtest_ma_strategy import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n回測被使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
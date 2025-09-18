#!/usr/bin/env python3
"""
測試字型顯示修正
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src/main/python"))

from utils.plot_config_english import setup_plotting

# 重新設置字型
setup_plotting()

# 創建簡單測試圖表
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 測試數據
x = np.arange(10)
y = np.random.randn(10).cumsum()

ax.plot(x, y, 'o-', label='Test Strategy')
ax.set_title('Font Display Test Chart')
ax.set_xlabel('Time Period')
ax.set_ylabel('Portfolio Value ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加一些文字標註
ax.text(5, y[5], 'Sample Text', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()

# 保存測試圖表
test_file = "output/font_test_chart.png"
Path("output").mkdir(exist_ok=True)
plt.savefig(test_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"測試圖表已保存到: {test_file}")
print("請檢查圖表中的文字是否正常顯示")
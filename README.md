# USA_VOOandQQQ_strategy

## 專案簡介

這是一個用於下載美股 ETF（VOO 和 QQQ）歷史資料並驗證交易策略的 Python AI/ML 專案。透過歷史數據回測，評估不同交易策略組合的績效表現。

## Quick Start

1. **Read CLAUDE.md first** - 包含 Claude Code 的基本規則
2. 遵循任務前的合規檢查清單
3. 使用正確的模組結構 `src/main/python/`
4. 每完成一個任務後提交

## 專案結構

這是一個 **AI/ML 專案**，採用完整的 MLOps 結構：

```
USA_VOOandQQQ_strategy/
├── src/main/python/       # 主要 Python 程式碼
│   ├── core/              # 核心交易演算法
│   ├── utils/             # 資料處理工具
│   ├── models/            # 策略模型定義
│   ├── services/          # ETF 資料服務
│   ├── api/               # 資料 API 介面
│   ├── training/          # 策略訓練腳本
│   ├── inference/         # 策略執行程式碼
│   └── evaluation/        # 績效評估指標
├── data/                  # ETF 資料管理
│   ├── raw/               # 原始 ETF 資料
│   ├── processed/         # 處理後的資料
│   └── temp/              # 暫存檔案
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # 資料探索
│   ├── experiments/       # 策略實驗
│   └── reports/           # 績效報告
├── models/                # 策略模型檔案
├── experiments/           # 實驗追蹤
├── output/                # 輸出檔案
└── logs/                  # 日誌檔案
```

## 主要功能

- **資料下載**：自動下載 VOO 和 QQQ 的歷史價格資料
- **策略回測**：在歷史資料上測試交易策略
- **績效分析**：計算收益率、夏普比率、最大回撤等指標
- **策略優化**：根據歷史表現優化策略參數
- **報告生成**：自動生成策略績效報告

## 開發指南

- **總是先搜尋**再建立新檔案
- **擴展現有**功能而不是重複建立
- **使用 Task agents** 處理超過 30 秒的操作
- **單一事實來源**原則
- **Python 專案**：所有程式碼放在 `src/main/python/` 下

## 安裝與設定

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (Linux/Mac)
source venv/bin/activate

# 安裝相依套件
pip install -r requirements.txt
```

## 常用指令

```bash
# 下載 ETF 資料
python src/main/python/services/data_downloader.py

# 執行回測
python src/main/python/core/backtest.py

# 生成報告
python src/main/python/evaluation/report_generator.py
```

## 授權

本專案使用 MIT 授權

---

🎯 Template by Chang Ho Chien | HC AI 說人話channel | v1.0.0
📺 Tutorial: https://youtu.be/8Q1bRZaHH24
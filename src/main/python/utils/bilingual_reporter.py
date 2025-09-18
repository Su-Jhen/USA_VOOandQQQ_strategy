"""
雙語報告生成器
支持中英文對照的報告輸出
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime


class BilingualReporter:
    """雙語報告生成器"""

    def __init__(self):
        self.bilingual_headers = {
            # CSV欄位標題
            'strategy_name': '策略名稱 / Strategy Name',
            'fast_period': '快線週期 / Fast MA Period',
            'slow_period': '慢線週期 / Slow MA Period',
            'qqq_weight_bear': '熊市QQQ權重 / Bear Market QQQ Weight',
            'use_slope_confirm': '斜率確認 / Slope Confirmation',
            'total_return': '總報酬率(%) / Total Return (%)',
            'cagr': '年化報酬率(%) / CAGR (%)',
            'volatility': '年化波動率(%) / Annual Volatility (%)',
            'sharpe_ratio': '夏普比率 / Sharpe Ratio',
            'max_drawdown': '最大回撤(%) / Max Drawdown (%)',
            'calmar_ratio': 'Calmar比率 / Calmar Ratio',
            'num_trades': '交易次數 / Number of Trades',
            'final_value': '最終價值($) / Final Value ($)',
            'bull_days': '牛市天數 / Bull Market Days',
            'bear_days': '熊市天數 / Bear Market Days',
            'signal_changes': '信號變化次數 / Signal Changes',

            # 權益曲線欄位
            'date': '日期 / Date',
            'total_value': '總資產價值($) / Total Value ($)',
            'cash': '現金($) / Cash ($)',
            'voo_weight': 'VOO權重 / VOO Weight',
            'qqq_weight': 'QQQ權重 / QQQ Weight',
            'market_state': '市場狀態 / Market State',
            'returns': '報酬率(%) / Returns (%)',

            # 交易記錄欄位
            'symbol': '標的 / Symbol',
            'action': '動作 / Action',
            'shares': '股數 / Shares',
            'price': '價格($) / Price ($)',
        }

        self.market_state_mapping = {
            'BULL': '牛市 / Bull',
            'BEAR': '熊市 / Bear',
            'NEUTRAL': '中性 / Neutral'
        }

        self.action_mapping = {
            'BUY': '買入 / Buy',
            'SELL': '賣出 / Sell'
        }

    def create_bilingual_csv_headers(self, df: pd.DataFrame, header_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        為DataFrame添加中英文對照標題

        Args:
            df: 原始DataFrame
            header_mapping: 自定義標題映射

        Returns:
            具有雙語標題的DataFrame
        """
        if header_mapping is None:
            header_mapping = self.bilingual_headers

        # 創建新的列名映射
        new_columns = {}
        for col in df.columns:
            if col in header_mapping:
                new_columns[col] = header_mapping[col]
            else:
                new_columns[col] = col  # 保持原名

        # 重命名列
        df_bilingual = df.copy()
        df_bilingual.rename(columns=new_columns, inplace=True)

        return df_bilingual

    def translate_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        翻譯分類數據的值

        Args:
            df: DataFrame

        Returns:
            翻譯後的DataFrame
        """
        df_translated = df.copy()

        # 翻譯市場狀態
        market_state_cols = [col for col in df.columns if 'market_state' in col.lower() or '市場狀態' in col]
        for col in market_state_cols:
            if col in df.columns:
                df_translated[col] = df_translated[col].map(self.market_state_mapping).fillna(df_translated[col])

        # 翻譯交易動作
        action_cols = [col for col in df.columns if 'action' in col.lower() or '動作' in col]
        for col in action_cols:
            if col in df.columns:
                df_translated[col] = df_translated[col].map(self.action_mapping).fillna(df_translated[col])

        # 翻譯布林值
        bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
        for col in bool_cols:
            df_translated[col] = df_translated[col].map({True: '是 / True', False: '否 / False'})

        return df_translated

    def generate_bilingual_text_report(self, metrics: Dict[str, Any], strategy_info: str = "") -> str:
        """
        生成雙語文字報告

        Args:
            metrics: 績效指標字典
            strategy_info: 策略信息

        Returns:
            雙語報告字符串
        """
        report = f"""
{'='*80}
回測績效報告 / Backtest Performance Report
{'='*80}

📊 基本資訊 / Basic Information
{'─'*60}
• 回測期間 / Backtest Period: {metrics.get('start_date', 'N/A')} 至 / to {metrics.get('end_date', 'N/A')}
• 初始資金 / Initial Capital: ${metrics.get('initial_capital', 0):,.0f}
• 年度加碼 / Annual Contribution: ${metrics.get('annual_contribution', 0):,.0f}
• 交易成本 / Transaction Cost: ${metrics.get('commission', 0):.0f} 每筆 / per trade

📈 績效指標 / Performance Metrics
{'─'*60}
• 總報酬率 / Total Return: {metrics.get('total_return', 0):.2f}%
• 年化報酬率 / CAGR: {metrics.get('cagr', 0):.2f}%
• 年化波動率 / Annual Volatility: {metrics.get('annual_volatility', 0):.2f}%
• 夏普比率 / Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
• 最大回撤 / Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
• Calmar比率 / Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}

💰 資金狀況 / Financial Status
{'─'*60}
• 最終資產價值 / Final Asset Value: ${metrics.get('final_value', 0):,.2f}
• 總交易次數 / Total Trades: {metrics.get('num_trades', 0)}

📋 策略參數 / Strategy Parameters
{'─'*60}
{strategy_info}

{'='*80}
報告生成時間 / Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        return report

    def create_performance_comparison_table(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        創建雙語績效比較表

        Args:
            results_dict: 結果字典

        Returns:
            雙語比較表DataFrame
        """
        comparison_data = []

        for strategy_name, result in results_dict.items():
            metrics = result.get('metrics', {})
            params = result.get('parameters', {})

            row = {
                'strategy_name': strategy_name,
                'fast_period': params.get('fast_period', 0),
                'slow_period': params.get('slow_period', 0),
                'qqq_weight_bear': f"{params.get('qqq_weight_bear', 0)*100:.0f}%",
                'use_slope_confirm': params.get('use_slope_confirm', False),
                'total_return': metrics.get('total_return', 0),
                'cagr': metrics.get('cagr', 0),
                'volatility': metrics.get('annual_volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'num_trades': metrics.get('num_trades', 0),
                'final_value': metrics.get('final_value', 0)
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 按夏普比率排序
        df = df.sort_values('sharpe_ratio', ascending=False)

        # 添加雙語標題
        df_bilingual = self.create_bilingual_csv_headers(df)
        df_bilingual = self.translate_categorical_values(df_bilingual)

        return df_bilingual

    def generate_strategy_summary(self, params: Dict[str, Any]) -> str:
        """
        生成雙語策略參數摘要

        Args:
            params: 策略參數字典

        Returns:
            雙語策略摘要
        """
        summary = f"""
MA交叉策略參數 / MA Crossover Strategy Parameters:
• 快線週期 / Fast MA Period: {params.get('fast_period', 0)} 日 / days
• 慢線週期 / Slow MA Period: {params.get('slow_period', 0)} 日 / days
• 牛市配置 / Bull Market Allocation: VOO {(1-params.get('qqq_weight_bull', 0))*100:.0f}% / QQQ {params.get('qqq_weight_bull', 0)*100:.0f}%
• 熊市配置 / Bear Market Allocation: VOO {(1-params.get('qqq_weight_bear', 0.5))*100:.0f}% / QQQ {params.get('qqq_weight_bear', 0.5)*100:.0f}%
• 再平衡閾值 / Rebalance Threshold: {params.get('rebalance_threshold', 0.05)*100:.0f}%
• 斜率確認 / Slope Confirmation: {'是 / Yes' if params.get('use_slope_confirm', False) else '否 / No'}
"""
        return summary

    def save_bilingual_reports(self,
                              results: Dict[str, Any],
                              output_dir: str = 'output',
                              timestamp: str = None) -> Dict[str, str]:
        """
        儲存所有雙語報告

        Args:
            results: 回測結果
            output_dir: 輸出目錄
            timestamp: 時間戳記

        Returns:
            儲存的檔案路徑字典
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = {}

        # 1. 權益曲線 (雙語標題)
        if 'equity_curve' in results:
            equity_file = output_path / f"equity_curve_bilingual_{timestamp}.csv"
            equity_df = self.create_bilingual_csv_headers(results['equity_curve'])
            equity_df = self.translate_categorical_values(equity_df)
            equity_df.to_csv(equity_file, index=True, encoding='utf-8-sig')
            saved_files['equity_curve'] = str(equity_file)

        # 2. 交易記錄 (雙語標題)
        if 'trades' in results:
            trades_file = output_path / f"trades_log_bilingual_{timestamp}.csv"
            trades_df = self.create_bilingual_csv_headers(results['trades'])
            trades_df = self.translate_categorical_values(trades_df)
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
            saved_files['trades'] = str(trades_file)

        # 3. 雙語文字報告
        if 'metrics' in results:
            report_file = output_path / f"backtest_report_bilingual_{timestamp}.txt"
            strategy_info = ""
            if 'strategy_params' in results:
                strategy_info = self.generate_strategy_summary(results['strategy_params'])

            report_content = self.generate_bilingual_text_report(results['metrics'], strategy_info)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            saved_files['report'] = str(report_file)

        return saved_files

    def create_top_strategies_report(self, comparison_df: pd.DataFrame, top_n: int = 10) -> str:
        """
        創建前N名策略的雙語報告

        Args:
            comparison_df: 策略比較DataFrame
            top_n: 顯示前N名

        Returns:
            前N名策略報告
        """
        top_strategies = comparison_df.head(top_n)

        report = f"""
🏆 前{top_n}名策略 / Top {top_n} Strategies
{'='*80}

排名 / Rank | 策略名稱 / Strategy Name | CAGR | 夏普比率 / Sharpe | 最大回撤 / Max DD
{'-'*80}
"""

        for idx, (_, row) in enumerate(top_strategies.iterrows(), 1):
            strategy_col = [col for col in row.index if '策略名稱' in col or 'strategy' in col.lower()][0]
            cagr_col = [col for col in row.index if 'cagr' in col.lower()][0]
            sharpe_col = [col for col in row.index if 'sharpe' in col.lower()][0]
            dd_col = [col for col in row.index if 'drawdown' in col.lower() or '回撤' in col][0]

            report += f"{idx:2d}     | {row[strategy_col]:<25} | {row[cagr_col]:5.2f}% | {row[sharpe_col]:6.3f} | {row[dd_col]:6.2f}%\n"

        report += f"\n{'-'*80}\n"
        return report


# 創建全局實例
bilingual_reporter = BilingualReporter()
"""
策略A：趨勢均線交叉法（MA Crossover）
使用VOO的移動平均線交叉來判斷市場趨勢
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MAStrategy:
    """MA交叉策略參數"""
    fast_period: int = 50      # 快線週期
    slow_period: int = 200     # 慢線週期
    qqq_weight_bull: float = 0.0   # 牛市QQQ權重（上漲期）
    qqq_weight_bear: float = 0.5   # 熊市QQQ權重（下跌期）
    rebalance_threshold: float = 0.05  # 再平衡閾值（5%）
    use_slope_confirm: bool = False    # 是否使用斜率確認
    slope_period: int = 20             # 斜率計算週期


class MAcrossoverStrategy:
    """MA交叉策略實作"""

    def __init__(self, params: MAStrategy = None):
        """
        初始化策略

        Args:
            params: 策略參數
        """
        self.params = params if params else MAStrategy()
        self.signals = pd.DataFrame()
        self.indicators = pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標

        Args:
            data: VOO價格數據

        Returns:
            包含指標的DataFrame
        """
        df = data.copy()

        # 計算移動平均線
        df['MA_fast'] = df['Close'].rolling(window=self.params.fast_period).mean()
        df['MA_slow'] = df['Close'].rolling(window=self.params.slow_period).mean()

        # 計算MA差值（用於判斷交叉強度）
        df['MA_diff'] = df['MA_fast'] - df['MA_slow']
        df['MA_diff_pct'] = (df['MA_diff'] / df['MA_slow']) * 100

        # 計算斜率（可選）
        if self.params.use_slope_confirm:
            # 慢線斜率
            df['MA_slow_slope'] = df['MA_slow'].diff(self.params.slope_period) / self.params.slope_period

        # 生成信號
        df['Signal'] = 0  # 0: 中性, 1: 牛市, -1: 熊市

        # 基本信號：MA交叉
        df.loc[df['MA_fast'] > df['MA_slow'], 'Signal'] = 1   # 金叉 -> 牛市
        df.loc[df['MA_fast'] < df['MA_slow'], 'Signal'] = -1  # 死叉 -> 熊市

        # 斜率確認（可選）
        if self.params.use_slope_confirm:
            # 只有當慢線也在下降時才確認熊市
            df.loc[(df['Signal'] == -1) & (df['MA_slow_slope'] >= 0), 'Signal'] = 0
            # 只有當慢線也在上升時才確認牛市
            df.loc[(df['Signal'] == 1) & (df['MA_slow_slope'] <= 0), 'Signal'] = 0

        # 儲存指標
        self.indicators = df
        return df

    def generate_signals(self, voo_data: pd.DataFrame, qqq_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號

        Args:
            voo_data: VOO數據
            qqq_data: QQQ數據

        Returns:
            交易信號DataFrame
        """
        # 計算指標
        indicators = self.calculate_indicators(voo_data)

        # 創建信號DataFrame
        signals = pd.DataFrame(index=indicators.index)
        signals['VOO_Close'] = voo_data['Close']
        signals['QQQ_Close'] = qqq_data['Close']

        # 市場狀態
        signals['Market_State'] = indicators['Signal'].map({
            1: 'BULL',
            -1: 'BEAR',
            0: 'NEUTRAL'
        })

        # 目標權重
        signals['QQQ_Target_Weight'] = indicators['Signal'].map({
            1: self.params.qqq_weight_bull,   # 牛市：0% QQQ
            -1: self.params.qqq_weight_bear,  # 熊市：50% QQQ
            0: 0.25  # 中性：25% QQQ
        })

        signals['VOO_Target_Weight'] = 1 - signals['QQQ_Target_Weight']

        # 添加指標值（用於分析）
        signals['MA_fast'] = indicators['MA_fast']
        signals['MA_slow'] = indicators['MA_slow']
        signals['MA_diff_pct'] = indicators['MA_diff_pct']

        # 標記信號變化點
        signals['Signal_Change'] = signals['Market_State'] != signals['Market_State'].shift(1)

        # 儲存信號
        self.signals = signals
        return signals

    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """
        判斷是否需要再平衡

        Args:
            current_weights: 當前權重
            target_weights: 目標權重

        Returns:
            是否需要再平衡
        """
        for symbol in ['VOO', 'QQQ']:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            if abs(current - target) > self.params.rebalance_threshold:
                return True
        return False

    def calculate_position_sizes(self, total_value: float,
                                target_weights: Dict[str, float],
                                prices: Dict[str, float]) -> Dict[str, int]:
        """
        計算目標持倉股數

        Args:
            total_value: 總資產價值
            target_weights: 目標權重
            prices: 當前價格

        Returns:
            各標的目標股數
        """
        positions = {}
        for symbol, weight in target_weights.items():
            if symbol != 'CASH':
                target_value = total_value * weight
                target_shares = int(target_value / prices[symbol])
                positions[symbol] = target_shares
        return positions

    def get_strategy_info(self) -> str:
        """獲取策略信息"""
        return f"""
MA交叉策略參數:
- 快線週期: {self.params.fast_period}日
- 慢線週期: {self.params.slow_period}日
- 牛市配置: VOO {(1-self.params.qqq_weight_bull)*100:.0f}% / QQQ {self.params.qqq_weight_bull*100:.0f}%
- 熊市配置: VOO {(1-self.params.qqq_weight_bear)*100:.0f}% / QQQ {self.params.qqq_weight_bear*100:.0f}%
- 再平衡閾值: {self.params.rebalance_threshold*100:.0f}%
- 斜率確認: {'是' if self.params.use_slope_confirm else '否'}
"""

    def analyze_signals(self) -> Dict:
        """分析信號統計"""
        if self.signals.empty:
            return {}

        # 信號統計
        signal_counts = self.signals['Market_State'].value_counts()
        signal_changes = self.signals['Signal_Change'].sum()

        # 計算各狀態持續時間
        state_durations = []
        current_state = None
        duration = 0

        for state in self.signals['Market_State']:
            if state != current_state:
                if current_state is not None:
                    state_durations.append(duration)
                current_state = state
                duration = 1
            else:
                duration += 1

        avg_duration = np.mean(state_durations) if state_durations else 0

        return {
            'bull_days': signal_counts.get('BULL', 0),
            'bear_days': signal_counts.get('BEAR', 0),
            'neutral_days': signal_counts.get('NEUTRAL', 0),
            'signal_changes': signal_changes,
            'avg_state_duration': avg_duration
        }


def create_ma_strategy_variants() -> Dict[str, MAStrategy]:
    """創建不同參數組合的MA策略"""
    variants = {
        'conservative': MAStrategy(
            fast_period=50,
            slow_period=200,
            qqq_weight_bear=0.3,
            use_slope_confirm=True
        ),
        'standard': MAStrategy(
            fast_period=50,
            slow_period=200,
            qqq_weight_bear=0.5,
            use_slope_confirm=False
        ),
        'aggressive': MAStrategy(
            fast_period=20,
            slow_period=100,
            qqq_weight_bear=0.7,
            use_slope_confirm=False
        ),
        'balanced': MAStrategy(
            fast_period=50,
            slow_period=150,
            qqq_weight_bear=0.4,
            use_slope_confirm=True
        )
    }
    return variants
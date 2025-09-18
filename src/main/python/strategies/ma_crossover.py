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

    # 進階優化參數
    use_slope_confirm: bool = False    # 是否使用斜率確認
    slope_period: int = 20             # 斜率計算週期

    use_crossover_filter: bool = False # 是否使用交叉強度過濾
    crossover_threshold: float = 0.01  # 交叉強度門檻（1%）

    use_duration_confirm: bool = False # 是否使用持續時間確認
    confirm_days: int = 3              # 信號確認天數

    # 動態權重調整
    use_dynamic_weight: bool = False   # 是否使用動態權重
    max_qqq_weight: float = 0.5       # QQQ最大權重限制


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

        # 應用進階優化過濾器
        df = self._apply_advanced_filters(df)

        return df

    def _apply_advanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        應用進階優化過濾器

        Args:
            df: 包含基本信號的DataFrame

        Returns:
            應用過濾後的DataFrame
        """
        # 1. 斜率確認過濾
        if self.params.use_slope_confirm:
            # 計算慢線斜率
            df['MA_slow_slope'] = df['MA_slow'].diff(self.params.slope_period) / self.params.slope_period

            # 雙重確認機制
            original_signal = df['Signal'].copy()

            # 下跌確認：MA_fast < MA_slow AND MA_slow斜率 < 0
            bear_confirmed = (original_signal == -1) & (df['MA_slow_slope'] < 0)

            # 上漲確認：MA_fast > MA_slow AND MA_slow斜率 > 0
            bull_confirmed = (original_signal == 1) & (df['MA_slow_slope'] > 0)

            # 重置信號，只保留確認的信號
            df['Signal'] = 0
            df.loc[bear_confirmed, 'Signal'] = -1
            df.loc[bull_confirmed, 'Signal'] = 1

        # 2. 交叉強度過濾
        if self.params.use_crossover_filter:
            # 計算交叉強度百分比
            df['MA_diff_pct'] = abs((df['MA_fast'] - df['MA_slow']) / df['MA_slow']) * 100

            # 只有當差距大於門檻時才認為是有效信號
            valid_crossover = df['MA_diff_pct'] >= (self.params.crossover_threshold * 100)
            df.loc[~valid_crossover, 'Signal'] = 0

        # 3. 持續時間確認過濾
        if self.params.use_duration_confirm:
            df = self._apply_duration_filter(df)

        return df

    def _apply_duration_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        應用持續時間確認過濾
        信號必須持續N天才被確認

        Args:
            df: DataFrame with signals

        Returns:
            過濾後的DataFrame
        """
        if len(df) < self.params.confirm_days:
            return df

        confirmed_signals = df['Signal'].copy()

        for i in range(self.params.confirm_days, len(df)):
            current_signal = df['Signal'].iloc[i]

            if current_signal != 0:  # 如果有信號
                # 檢查前N天是否都是同樣的信號
                lookback_signals = df['Signal'].iloc[i-self.params.confirm_days+1:i+1]

                if not all(lookback_signals == current_signal):
                    # 如果不是連續N天同樣信號，則取消此信號
                    confirmed_signals.iloc[i] = 0

        df['Signal'] = confirmed_signals
        return df

    def generate_signals(self, voo_data: pd.DataFrame, qqq_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        生成交易信號（兼容版本）

        Args:
            voo_data: VOO價格數據
            qqq_data: QQQ價格數據（可選，用於兼容性）

        Returns:
            包含信號的DataFrame
        """
        # 計算指標
        indicators = self.calculate_indicators(voo_data)

        # 創建信號DataFrame
        signals = pd.DataFrame(index=indicators.index)
        signals['VOO_Close'] = voo_data['Close']
        if qqq_data is not None:
            signals['QQQ_Close'] = qqq_data['Close']

        # 市場狀態
        signals['Market_State'] = indicators['Signal'].map({
            1: 'BULL',
            -1: 'BEAR',
            0: 'NEUTRAL'
        })

        # 目標權重（使用動態權重）
        if self.params.use_dynamic_weight:
            qqq_weights = self._calculate_dynamic_qqq_weight(indicators)
        else:
            qqq_weights = pd.Series([self.params.qqq_weight_bear] * len(indicators), index=indicators.index)

        signals['QQQ_Target_Weight'] = indicators['Signal'].map({
            1: self.params.qqq_weight_bull,   # 牛市：0% QQQ
            0: 0.25  # 中性：25% QQQ
        })

        # 對熊市使用動態權重
        bear_mask = indicators['Signal'] == -1
        signals.loc[bear_mask, 'QQQ_Target_Weight'] = qqq_weights[bear_mask]

        signals['VOO_Target_Weight'] = 1 - signals['QQQ_Target_Weight']

        # 添加指標值（用於分析）
        signals['MA_fast'] = indicators['MA_fast']
        signals['MA_slow'] = indicators['MA_slow']
        if 'MA_diff_pct' in indicators.columns:
            signals['MA_diff_pct'] = indicators['MA_diff_pct']

        # 標記信號變化點
        signals['Signal_Change'] = signals['Market_State'] != signals['Market_State'].shift(1)

        # 儲存信號
        self.signals = signals
        return signals

    def _calculate_dynamic_qqq_weight(self, df: pd.DataFrame) -> pd.Series:
        """
        計算動態QQQ權重

        Args:
            df: 包含指標的DataFrame

        Returns:
            動態QQQ權重Series
        """
        if not self.params.use_dynamic_weight:
            return pd.Series([self.params.qqq_weight_bear] * len(df), index=df.index)

        # 基礎權重
        base_weight = self.params.qqq_weight_bear

        # 根據下跌深度調整（如果有價格數據）
        if 'Close' in df.columns:
            # 計算從最高點的回撤
            rolling_max = df['Close'].rolling(window=252, min_periods=1).max()  # 一年滾動最高
            drawdown = (rolling_max - df['Close']) / rolling_max

            # 根據回撤深度調整權重
            # 回撤越深，QQQ權重越高（在熊市中買入更多便宜的QQQ）
            adjustment = drawdown * 0.5  # 最多增加50%的額外權重
            dynamic_weight = np.minimum(
                base_weight + adjustment,
                self.params.max_qqq_weight
            )
        else:
            dynamic_weight = pd.Series([base_weight] * len(df), index=df.index)

        return dynamic_weight

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
    """創建不同參數組合的MA策略（基本變體）"""
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


def create_ma_parameter_matrix() -> Dict[str, MAStrategy]:
    """
    根據策略計畫創建完整的MA參數矩陣
    實作策略計畫中的所有參數組合
    """
    variants = {}

    # 策略計畫中的MA參數組合
    ma_combinations = [
        (10, 30, "超短期"),    # 敏感度高，交易頻繁
        (20, 50, "短期"),      # 短線交易常用
        (50, 100, "中期"),     # 平衡版本
        (50, 200, "經典"),     # 黃金交叉/死亡交叉
        (100, 200, "長期"),    # 長線趨勢
    ]

    # QQQ熊市權重選項
    qqq_weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # 斜率確認選項
    slope_confirm_options = [True, False]

    for fast, slow, period_type in ma_combinations:
        for qqq_weight in qqq_weights:
            for use_slope in slope_confirm_options:
                # 創建策略名稱
                slope_suffix = "_slope" if use_slope else ""
                name = f"{period_type}_{fast}_{slow}_QQQ{int(qqq_weight*100)}{slope_suffix}"

                variants[name] = MAStrategy(
                    fast_period=fast,
                    slow_period=slow,
                    qqq_weight_bear=qqq_weight,
                    qqq_weight_bull=0.0,  # 牛市始終0% QQQ
                    use_slope_confirm=use_slope,
                    rebalance_threshold=0.05
                )

    return variants


def create_comprehensive_parameter_scan() -> Dict[str, MAStrategy]:
    """
    創建更詳細的參數掃描
    用於尋找最優參數組合
    """
    variants = {}

    # 擴展的MA參數範圍
    fast_periods = [5, 10, 15, 20, 30, 40, 50, 60]
    slow_periods = [30, 50, 100, 150, 200, 250]

    # QQQ權重範圍（每10%一個級距）
    qqq_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # 只測試有效的MA組合（fast < slow）
    valid_combinations = []
    for fast in fast_periods:
        for slow in slow_periods:
            if fast < slow and slow >= fast * 1.5:  # 確保slow至少是fast的1.5倍
                valid_combinations.append((fast, slow))

    # 為避免組合數過多，我們採用抽樣策略
    # 重點測試一些關鍵組合
    key_combinations = [
        (10, 30), (10, 50), (15, 50), (20, 50), (20, 100),
        (30, 100), (40, 100), (50, 100), (50, 150), (50, 200),
        (60, 150), (60, 200)
    ]

    for fast, slow in key_combinations:
        for qqq_weight in [0.3, 0.5, 0.7]:  # 只測試關鍵權重
            for use_slope in [True, False]:
                slope_suffix = "_slope" if use_slope else ""
                name = f"scan_{fast}_{slow}_QQQ{int(qqq_weight*100)}{slope_suffix}"

                variants[name] = MAStrategy(
                    fast_period=fast,
                    slow_period=slow,
                    qqq_weight_bear=qqq_weight,
                    qqq_weight_bull=0.0,
                    use_slope_confirm=use_slope,
                    rebalance_threshold=0.05
                )

    return variants
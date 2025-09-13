"""
回測引擎核心模組
處理策略執行、部位管理、績效計算
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Position:
    """持倉記錄"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        """市值"""
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """未實現損益"""
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def return_pct(self) -> float:
        """報酬率"""
        if self.avg_cost == 0:
            return 0
        return (self.current_price / self.avg_cost - 1) * 100


@dataclass
class Trade:
    """交易記錄"""
    date: pd.Timestamp
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    commission: float

    @property
    def total_cost(self) -> float:
        """總成本（含手續費）"""
        if self.action == 'BUY':
            return self.shares * self.price + self.commission
        else:
            return self.shares * self.price - self.commission


class Portfolio:
    """投資組合管理"""

    def __init__(self, initial_capital: float = 10000, commission: float = 3.0):
        """
        初始化投資組合

        Args:
            initial_capital: 初始資金
            commission: 每筆交易手續費
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def buy(self, symbol: str, shares: float, price: float, date: pd.Timestamp) -> bool:
        """
        買入股票

        Args:
            symbol: 股票代碼
            shares: 股數
            price: 價格
            date: 交易日期

        Returns:
            是否成功執行
        """
        total_cost = shares * price + self.commission

        if self.cash < total_cost:
            # 資金不足，計算能買的最大股數
            max_shares = int((self.cash - self.commission) / price)
            if max_shares <= 0:
                return False
            shares = max_shares
            total_cost = shares * price + self.commission

        # 執行交易
        self.cash -= total_cost

        # 更新持倉
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            avg_cost = (pos.shares * pos.avg_cost + shares * price) / total_shares
            self.positions[symbol] = Position(symbol, total_shares, avg_cost, price)
        else:
            self.positions[symbol] = Position(symbol, shares, price, price)

        # 記錄交易
        self.trades.append(Trade(date, symbol, 'BUY', shares, price, self.commission))
        return True

    def sell(self, symbol: str, shares: float, price: float, date: pd.Timestamp) -> bool:
        """
        賣出股票

        Args:
            symbol: 股票代碼
            shares: 股數
            price: 價格
            date: 交易日期

        Returns:
            是否成功執行
        """
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        if pos.shares < shares:
            shares = pos.shares  # 賣出全部

        # 執行交易
        self.cash += shares * price - self.commission

        # 更新持倉
        pos.shares -= shares
        if pos.shares <= 0:
            del self.positions[symbol]

        # 記錄交易
        self.trades.append(Trade(date, symbol, 'SELL', shares, price, self.commission))
        return True

    def update_prices(self, prices: Dict[str, float]):
        """更新持倉價格"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]

    @property
    def total_value(self) -> float:
        """總資產價值"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def returns(self) -> float:
        """總報酬率"""
        return (self.total_value / self.initial_capital - 1) * 100

    def get_position_weights(self) -> Dict[str, float]:
        """獲取持倉權重"""
        total = self.total_value
        if total == 0:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            weights[symbol] = pos.market_value / total
        weights['CASH'] = self.cash / total
        return weights


class BacktestEngine:
    """回測引擎"""

    def __init__(self,
                 start_date: str = '2010-09-09',
                 end_date: str = '2025-09-12',
                 initial_capital: float = 10000,
                 annual_contribution: float = 3000,
                 commission: float = 3.0):
        """
        初始化回測引擎

        Args:
            start_date: 開始日期
            end_date: 結束日期
            initial_capital: 初始資金
            annual_contribution: 年度加碼金額
            commission: 手續費
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.annual_contribution = annual_contribution
        self.commission = commission

        # 數據容器
        self.data: Dict[str, pd.DataFrame] = {}
        self.signals: pd.DataFrame = None
        self.portfolio = Portfolio(initial_capital, commission)
        self.results: Dict = {}

    def load_data(self, symbols: List[str] = ['VOO', 'QQQ']):
        """
        載入歷史數據

        Args:
            symbols: 股票代碼列表
        """
        for symbol in symbols:
            try:
                # 載入CSV數據
                filepath = f'data/raw/{symbol}_complete.csv'
                df = pd.read_csv(filepath)
                # 處理日期（去除時區信息）
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                df.set_index('Date', inplace=True)

                # 過濾日期範圍
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

                # 儲存數據
                self.data[symbol] = df
                print(f"✅ 載入 {symbol} 數據: {len(df)} 筆")

            except Exception as e:
                print(f"❌ 載入 {symbol} 失敗: {str(e)}")

    def get_annual_contribution_dates(self) -> List[pd.Timestamp]:
        """獲取年度加碼日期（每年第一個交易日）"""
        if not self.data:
            return []

        # 使用VOO的交易日期
        dates = self.data['VOO'].index
        contribution_dates = []

        # 第一年（2011）投入初始資金
        # 從2012開始每年加碼
        for year in range(2012, self.end_date.year + 1):
            year_dates = dates[dates.year == year]
            if len(year_dates) > 0:
                contribution_dates.append(year_dates[0])

        return contribution_dates

    def calculate_metrics(self) -> Dict:
        """計算績效指標"""
        if not hasattr(self, 'equity_curve') or self.equity_curve.empty:
            return {}

        equity = self.equity_curve['total_value']
        returns = self.equity_curve['returns']

        # 基本指標
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

        # 年化報酬率（CAGR）
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (np.power(equity.iloc[-1] / equity.iloc[0], 1/years) - 1) * 100

        # 日報酬率
        daily_returns = equity.pct_change().dropna()

        # 年化波動率
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100

        # 夏普比率（假設無風險利率為3%）
        risk_free_rate = 0.03
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Calmar比率
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # 交易統計
        num_trades = len(self.portfolio.trades)

        return {
            'total_return': round(total_return, 2),
            'cagr': round(cagr, 2),
            'annual_volatility': round(annual_volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(calmar_ratio, 3),
            'num_trades': num_trades,
            'final_value': round(equity.iloc[-1], 2)
        }

    def generate_report(self) -> str:
        """生成回測報告"""
        metrics = self.calculate_metrics()

        report = f"""
{'='*60}
回測績效報告
{'='*60}

📊 基本資訊
- 回測期間: {self.start_date.date()} 至 {self.end_date.date()}
- 初始資金: ${self.initial_capital:,.0f}
- 年度加碼: ${self.annual_contribution:,.0f}
- 交易成本: ${self.commission:.0f}/筆

📈 績效指標
- 總報酬率: {metrics.get('total_return', 0):.2f}%
- 年化報酬率(CAGR): {metrics.get('cagr', 0):.2f}%
- 年化波動率: {metrics.get('annual_volatility', 0):.2f}%
- 夏普比率: {metrics.get('sharpe_ratio', 0):.3f}
- 最大回撤: {metrics.get('max_drawdown', 0):.2f}%
- Calmar比率: {metrics.get('calmar_ratio', 0):.3f}

💰 資金狀況
- 最終資產價值: ${metrics.get('final_value', 0):,.2f}
- 總交易次數: {metrics.get('num_trades', 0)}

{'='*60}
"""
        return report
import logging
from typing import Dict, List

from .config import Config

class PortfolioManager:
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_portfolio_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate portfolio-level metrics"""
        metrics = {}
        
        # Diversification score (simple example: count unique symbols)
        unique_symbols = len(set(s['symbol'] for s in signals))
        metrics['diversification_score'] = unique_symbols / len(self.config.SYMBOLS) if self.config.SYMBOLS else 0
        
        # Count buy/sell signals
        metrics['buy_signals'] = sum(1 for s in signals if 'BUY' in s['type'])
        metrics['sell_signals'] = sum(1 for s in signals if 'SELL' in s['type'])
        
        return metrics

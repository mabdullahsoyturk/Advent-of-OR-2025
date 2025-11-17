import os

import nextmv
import pandas as pd
from instance_manager import InstanceManager
from portfolio_optimizer import optimize_portfolio


def solve_optimization(df_segments: pd.DataFrame, df_assets: pd.DataFrame, 
                       df_correlation_matrix: pd.DataFrame, risk_weight_limit: float, 
                       consider_risk:bool = True, z_score: float = 1.96, profit_weight: float=None,
                       options: nextmv.Options=None) ->pd.DataFrame:

    # Load instance from CSV
    manager = InstanceManager(instance_name="")
    original_portfolio = manager.from_csv(df_segments, df_assets, df_correlation_matrix)

    # Optimize the portfolio with a specified risk weight limit
    investment = optimize_portfolio(
        original_portfolio, 
        risk_weight_limit, 
        consider_risk=consider_risk, 
        z_score=z_score, 
        profit_weight=profit_weight, 
        options=options,
    )
    optimized_portfolio = manager.new_portfolio(original_portfolio, investment)
    
    # Write the new portfolio to tables
    df_segments, df_assets = optimized_portfolio.to_tables(extract=True)

    return df_segments, df_assets

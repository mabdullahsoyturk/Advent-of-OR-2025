from typing import Dict, Tuple
from domain import Asset, Portfolio, Segment
import pandas as pd
import os


class InstanceManager:
    def __init__(self, instance_name="test"):
        self.instance_name = instance_name
    
    def from_csv(self, df_segments:pd.DataFrame, df_assets:pd.DataFrame, df_correlation_matrix: pd.DataFrame) -> Portfolio:
        # df = pd.read_csv(os.path.join(folder_path, "segments.csv"))
        assets = {}
        for segment_id, row in df_segments.iterrows():
            asset_id = row['asset']
            exposure = row['exposure']
            risk_weight = row['risk_weight']
            profitability = row['average_profitability']
            rel_sell_cost = row.get('rel_sell_cost', 0.0)
            rel_origination_cost = row.get('rel_origination_cost', 0.0)

            segment = Segment(segment_id, asset_id, exposure, profitability, risk_weight, rel_sell_cost, rel_origination_cost)

            if asset_id not in assets:
                assets[asset_id] = Asset(asset_id, {segment_id: segment}, exposure, exposure * profitability, exposure * risk_weight, risk_weight)
            else:
                assets[asset_id].add_segment(segment)

        # Update asset level information
        # df = pd.read_csv(os.path.join(folder_path, "assets.csv"))
        for asset_id, row in df_assets.iterrows():
            min_rel_exposure = 1-row['max_exposure_decrease']
            max_rel_exposure = 1+row['max_exposure_increase']
            stdev = row.get('stdev_profitability', 0.0)

            if asset_id in assets:
                assets[asset_id].min_rel_exposure = min_rel_exposure
                assets[asset_id].max_rel_exposure = max_rel_exposure
                assets[asset_id].profit_stdev = stdev

        return Portfolio(f"{self.instance_name}_input_portfolio", assets, df_correlation_matrix)

    def new_portfolio(self, portfolio: Portfolio, new_distribution: Dict[Tuple[str, str], float]) -> Portfolio:
        new_assets = {}
        portfolio_id = f"{self.instance_name}_optimized_portfolio"
        for asset in portfolio.assets.values():
            for segment in asset.segments.values():
                new_segment = Segment(segment.segment_id, segment.asset, segment.exposure, segment.profitability,
                                        segment.risk_weight, segment.rel_sell_cost, segment.rel_origination_cost)
                new_segment.exposure = round(new_distribution[(asset.asset_id, segment.segment_id)] * new_segment.exposure)
            
                if asset.asset_id not in new_assets:
                    new_assets[asset.asset_id] = Asset(asset.asset_id, {new_segment.segment_id: new_segment}, new_segment.exposure, 
                                                    new_segment.exposure * new_segment.profitability, new_segment.exposure * new_segment.risk_weight, 
                                                    new_segment.risk_weight, asset.min_rel_exposure, asset.max_rel_exposure)
                else:
                    new_assets[asset.asset_id].add_segment(new_segment)

        return Portfolio(portfolio_id, new_assets)
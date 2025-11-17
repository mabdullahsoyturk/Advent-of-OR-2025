from dataclasses import dataclass, field
from typing import Dict, Tuple
import pandas as pd
import os

@dataclass
class Segment:
    """
    Class that contains the important attribute of each subsegment
    """
    segment_id: str
    asset: str
    exposure: float
    profitability: float
    risk_weight: float
    rel_sell_cost: float
    rel_origination_cost: float

@dataclass
class Asset:
    """
    Class that contains the important attributes of an asset
    """
    asset_id: str
    segments: Dict[str, Segment] = field(default_factory=dict)
    total_exposure: float = 0.0
    total_profit: float = 0.0
    total_risk_weighted_assets: float = 0.0
    average_risk_weight: float = 0.0
    min_rel_exposure: float = 0.5  # Minimum allowable exposure for the asset
    max_rel_exposure: float = 1.5  # Maximum allowable exposure for the asset
    profit_stdev: float = 0.0  # Standard deviation of the asset's profitability

    def add_segment(self, segment: Segment):
        """
        Add a segment to the asset and update aggregate metrics
        """
        self.segments[segment.segment_id] = segment
        self.total_exposure += segment.exposure
        self.total_profit += segment.profitability * segment.exposure
        self.total_risk_weighted_assets += segment.exposure * segment.risk_weight
        self.average_risk_weight = self.total_risk_weighted_assets / self.total_exposure if self.total_exposure > 0 else 0

@dataclass
class Portfolio:
    """
    Class that contains the important attributes of a portfolio
    """
    portfolio_id: str
    assets: Dict[str, Asset] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_exposure: float = 0.0
    total_profit: float = 0.0
    total_risk_weighted_assets: float = 0.0
    average_risk_weight: float = 0.0

    def __post_init__(self):
        """
        Add an asset to the instance and update aggregate metrics
        """
        for asset in self.assets.values():
            self.total_exposure += asset.total_exposure
            self.total_profit += asset.total_profit
            self.total_risk_weighted_assets += asset.total_risk_weighted_assets
            self.average_risk_weight = self.total_risk_weighted_assets / self.total_exposure if self.total_exposure > 0 else 0

    def to_tables(self, extract=False, folder_path="results") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export the portfolio data to CSV format
        """
        os.makedirs(folder_path, exist_ok=True)
        segments_data = []
        for asset in self.assets.values():
            for segment in asset.segments.values():
                segments_data.append({
                    'asset': asset.asset_id,
                    'segment_id': segment.segment_id,
                    'exposure': segment.exposure,
                    'profitability': segment.profitability,
                    'risk_weight': segment.risk_weight,
                    'rel_sell_cost': segment.rel_sell_cost,
                    'rel_origination_cost': segment.rel_origination_cost
                })

        df_segments = pd.DataFrame(segments_data)
        df_segments.set_index('segment_id')

        assets_data = []
        for asset in self.assets.values():
            assets_data.append({
                'asset': asset.asset_id,
                'total_exposure': asset.total_exposure,
                'total_profit': asset.total_profit,
                'average_risk_weight': asset.average_risk_weight,
                'min_rel_exposure': asset.min_rel_exposure,
                'max_rel_exposure': asset.max_rel_exposure
            })

        df_assets = pd.DataFrame(assets_data)

        if extract:
            df_segments.to_csv(f"{folder_path}/segments_{self.portfolio_id}.csv", index=False)
            df_assets.to_csv(f"{folder_path}/assets_{self.portfolio_id}.csv", index=False)
        return df_segments, df_assets

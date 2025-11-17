# Quick start template for Python based Xpress Insight applications.
# Copyright (c) 2020-2023 Fair Isaac Corporation. All rights reserved.

import math
import os
import sys

import nextmv
import pandas as pd
import xpressinsight as xi
from entry_point import solve_optimization
from scipy.stats import norm

manifest = nextmv.Manifest.from_yaml(dirpath=os.path.join(os.path.dirname(__file__), ".."))
options = manifest.extract_options()

@xi.AppConfig(name="Portfolio Rebalancing", version=xi.AppVersion(1, 0, 0), raise_attach_exceptions=True)
class InsightApp(xi.AppBase):
    # Inputs need to be initialized in the load mode(s).
    # Entities are inputs if the manage attribute is xi.Manage.INPUT. This is the default, i.e., it can be omitted.

    # Maximum Portfolio Risk Weight
    MaxPortfolioRiskWeight: xi.types.Scalar(0.5, dtype=xi.real)
    ProfitWeight: xi.types.Scalar(-1.0, dtype=xi.real)
    InitialExposure: xi.types.Scalar(0.0, dtype=xi.real)
    ConsiderRisk: xi.types.Scalar(True, dtype=xi.boolean)
    ConfidenceLevel: xi.types.Scalar(0.95, format="0.0%", dtype=xi.real)

    # Indices of the tables/dataframes that will be used
    SegmentIds: xi.types.Index(dtype=xi.string, alias="Segments")
    AssetIds: xi.types.Index(dtype=xi.string, alias="Assets")
    SolutionIds: xi.types.Index(dtype=xi.string, alias="Assets")

    # Dataframes that will be used in the application
    Segments: xi.types.DataFrame(index="SegmentIds", columns=[
        xi.types.Column("asset", dtype=xi.string, alias="Asset Name"),
        xi.types.Column("exposure", dtype=xi.integer, alias="Initial Exposure"),
        xi.types.Column("average_profitability", dtype=xi.real, format="0.0%", alias="Expected Profitability"),
        xi.types.Column("risk_weight", dtype=xi.real, format="0.0%", alias="Risk Weight"),
        xi.types.Column("rel_sell_cost", dtype=xi.real, format="0.0%", alias="Selling Cost"),
        xi.types.Column("rel_origination_cost", dtype=xi.real, format="0.0%", alias="Origination Cost"),
        xi.types.Column("optimized_exposure", dtype=xi.integer, alias="Optimized Exposure", manage=xi.Manage.RESULT),
        xi.types.Column("exposure_change", dtype=xi.integer, alias="Exposure Change", manage=xi.Manage.RESULT)
    ])

    Assets: xi.types.DataFrame(index="AssetIds", columns=[
        xi.types.Column("max_exposure_decrease", dtype=xi.real, format="0.0%", alias="Maximum Decrease"),
        xi.types.Column("max_exposure_increase", dtype=xi.real, format="0.0%", alias="Maximum Increase"),
        xi.types.Column("stdev_profitability", dtype=xi.real, format="0.0%", alias="Profitability Standard Deviation"),
        xi.types.Column("exposure", dtype=xi.integer, alias="Original Exposure"),
        xi.types.Column("risk_weight", dtype=xi.real, alias="Original Risk Weight"),
        xi.types.Column("optimized_exposure", dtype=xi.integer, alias="Optimized Exposure", manage=xi.Manage.RESULT),
        xi.types.Column("average_risk_weight", dtype=xi.real, alias="Optimized Average Risk Weight", manage=xi.Manage.RESULT),
    ])

    CorrelationMatrix:xi.types.DataFrame(index="AssetIds", columns=[
        xi.types.Column("Retail_Mortgage", dtype=xi.real, format="0.0%", alias="Retail Mortgage"),
        xi.types.Column("Retail_Revolving", dtype=xi.real, format="0.0%", alias="Retail Revolving"),
        xi.types.Column("Retail_Other", dtype=xi.real, format="0.0%", alias="Retail Other"),
    ])

    # Portfolio level Results
    NetProfit: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)
    ExpectedProfit: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)
    TransactionCosts: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)
    ARW: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)
    OptimizedExposure: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)
    PortfolioProfitDownside: xi.types.Scalar(0.0, dtype=xi.real, manage=xi.Manage.RESULT)

    Portfolio_Result: xi.types.DataFrame(index="SolutionIds", columns=[
        xi.types.Column("net_profit", dtype=xi.real, format="$,000.00", alias="Net Profit", manage=xi.Manage.RESULT),
        xi.types.Column("expected_profit", dtype=xi.real, format="$,000.00", alias="Expected Profit", manage=xi.Manage.RESULT),
        xi.types.Column("transaction_costs", dtype=xi.real, format="$,000.00", alias="Transaction Costs", manage=xi.Manage.RESULT),
        xi.types.Column("average_risk_weight", dtype=xi.real, alias="Average Risk Weight", manage=xi.Manage.RESULT),
        xi.types.Column("initial_exposure", format="$,000.00", dtype=xi.real, alias="Initial Exposure", manage=xi.Manage.RESULT),
        xi.types.Column("optimized_exposure", dtype=xi.real, format="$,000.00", alias="Optimized Exposure", manage=xi.Manage.RESULT),
        xi.types.Column("downside", dtype=xi.real, format="$,000.00", alias="Profitability Downside", manage=xi.Manage.RESULT),
    ])


    @xi.ExecModeLoad(descr="Loads input data.")
    def load(self):
        # Scenario is being 'loaded' through Xpress Insight.
        # Insight automatically populates the parameters with the values from the UI.
        # Input entities will be captured and stored in the scenario.
        print("Loading data.")
        # Load Segment information
        self.Segments = pd.read_csv(self.insight.get_attach_by_tag('segments-file').filename, index_col=['segment_id'])
        self.SegmentIds = self.Segments.index
        self.InitialExposure = float(self.Segments['exposure'].sum())

        self.SolutionIds = pd.Index(['Optimized Portfolio'])

        # Get the original exposure at the asset level
        temp_segments = self.Segments.copy()
        temp_segments['rwa'] = temp_segments['exposure']*temp_segments['risk_weight']
        temp_assets = temp_segments.groupby(['asset']).agg({'exposure':'sum', 'rwa':'sum'})
        # Load Asset information
        self.Assets = pd.read_csv(self.insight.get_attach_by_tag('assets-file').filename, index_col=['asset'])
        self.AssetIds = self.Assets.index
        self.Assets['exposure'] = temp_assets['exposure']
        self.Assets['risk_weight'] = temp_assets['rwa']/temp_assets['exposure']

        # Load Correlation Matrix information
        self.CorrelationMatrix = pd.read_csv(self.insight.get_attach_by_tag('correlation-file').filename, index_col=['asset'])
        print("\nLoad mode finished.")

    @xi.ExecModeRun(descr="Takes input data and uses it to compute the results.")
    def run(self):
        # Result entities will be captured and stored in the scenario.
        print('Scenario:', self.insight.scenario_name)
        # Insight automatically populates the parameters and inputs with the values from the UI.
        # If not considering risk or if considering it with hierarchical optimization, then no need for a profit weight
        if self.ConsiderRisk==False:
            self.ProfitWeight = -1

        z_score = norm.ppf(self.ConfidenceLevel)
        df_result, df_assets = solve_optimization(
            self.Segments, 
            self.Assets, 
            self.CorrelationMatrix, 
            self.MaxPortfolioRiskWeight, 
            self.ConsiderRisk, 
            z_score, 
            self.ProfitWeight, 
            options=options,
        )

        # Update the results dataframes with the optimization solution
        self.Segments['optimized_exposure'] = pd.Series(list(df_result['exposure']), index=self.SegmentIds)
        self.Segments['exposure_change'] = (self.Segments['optimized_exposure'] - self.Segments['exposure'])
        self.Assets['optimized_exposure'] = pd.Series(list(df_assets['total_exposure']), index=self.AssetIds)
        self.Assets['average_risk_weight'] = pd.Series(list(df_assets['average_risk_weight']), index=self.AssetIds)
        self.Assets['portfolio_ratio'] = self.Assets['optimized_exposure']/self.Assets['optimized_exposure'].sum()

        # Calculate the different portfolio level KPIs
        self.ExpectedProfit = (self.Segments['optimized_exposure']*self.Segments['average_profitability']).sum()
        self.TransactionCosts = (self.Segments.loc[self.Segments['exposure_change'] > 0, 'exposure_change'] * self.Segments.loc[self.Segments['exposure_change'] > 0, 'rel_origination_cost']).sum()\
            -(self.Segments.loc[self.Segments['exposure_change'] < 0, 'exposure_change'] * self.Segments.loc[self.Segments['exposure_change'] < 0, 'rel_sell_cost']).sum()
        self.OptimizedExposure = float(self.Segments['optimized_exposure'].sum())
        self.ARW = (self.Segments['optimized_exposure']*self.Segments['risk_weight']).sum()/self.OptimizedExposure
        self.NetProfit = self.ExpectedProfit - self.TransactionCosts
        self.PortfolioProfitDownside = z_score * self.ExpectedProfit * math.sqrt(sum([self.Assets.loc[asset_i, 'stdev_profitability']*self.Assets.loc[asset_j, 'stdev_profitability']*
                                     self.Assets.loc[asset_i, 'portfolio_ratio'] * self.Assets.loc[asset_j, 'portfolio_ratio'] *
                                     self.CorrelationMatrix.loc[asset_i, asset_j] for asset_i in self.Assets.index for asset_j in self.Assets.index]))


        solution = [{
            'net_profit': self.NetProfit,
            'expected_profit': self.ExpectedProfit,
            'transaction_costs': self.TransactionCosts,
            'average_risk_weight': self.ARW,
            'initial_exposure': self.InitialExposure,
            'optimized_exposure': self.OptimizedExposure,
            'downside': self.PortfolioProfitDownside
        }]
        self.Portfolio_Result = pd.DataFrame(solution, index=self.SolutionIds)
        
        stats = nextmv.Statistics(
            result=nextmv.ResultStatistics(
                custom=solution[0],
            ),
        )
        output = nextmv.Output(
            options=options,
            output_format=nextmv.OutputFormat.MULTI_FILE,
            solution_files=[
                nextmv.csv_solution_file(name="segments", data=df_result.to_dict(orient="records")),
                nextmv.csv_solution_file(name="assets", data=df_assets.to_dict(orient="records")),
            ],
            statistics=stats,
        )
        nextmv.write(output=output)

        print(f"Expected Profit of {self.ExpectedProfit} and Transaction Costs {self.TransactionCosts} with a total profit improvement of {self.NetProfit} average risk weight of {self.ARW} Portfolio Downside {self.PortfolioProfitDownside} new exposure of {self.OptimizedExposure} from initial exposure {self.InitialExposure}")


if __name__ == "__main__":
    # When the application is run in test mode (i.e., outside of Xpress Insight),
    # first initialize the test environment, then execute the load and run modes.
    app = xi.create_app(InsightApp)
    sys.exit(app.call_exec_modes(["LOAD", "RUN"]))
    


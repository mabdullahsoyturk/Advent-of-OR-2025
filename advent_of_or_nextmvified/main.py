import os
import sys

import gamspy as gp
import nextmv
import pandas as pd
from scipy.stats import norm


def main(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    correlation_df: pd.DataFrame,
    options: nextmv.Options,
) -> nextmv.Output:
    with gp.Container():
        A = gp.Set(
            description="Set of all assets in the portfolio",
            records=assets["asset"].unique(),
        )
        S = gp.Set(description="Set of segments", records=segments["segment_id"])
        AS = gp.Set(domain=[A, S], records=segments[["asset", "segment_id"]])
        J = gp.Alias(alias_with=A)

        profitability = gp.Parameter(
            domain=[A, S],
            description="Expected profitability for asset a, segment s",
            records=segments[["asset", "segment_id", "average_profitability"]],
        )
        risk_weight_limit = gp.Parameter(
            description="Maximum allowable average risk weight at the portfolio level",
            records=options.risk_weight_limit,
        )
        z_score = gp.Parameter(
            records=norm.ppf(options.confidence_interval),
            description="Z-score for risk calculation (default: corresponding to 95 percent confidence level in one tail test)",
        )
        lo_a = gp.Parameter(
            domain=A,
            description="Lower bound for asset `a` relative exposure",
            records=assets[["asset", "max_exposure_decrease"]],
        )
        up_a = gp.Parameter(
            domain=A,
            description="Upper bound for asset `a` relative exposure",
            records=assets[["asset", "max_exposure_increase"]],
        )
        rel_origination_cost = gp.Parameter(
            domain=[A, S],
            description="Per unit cost of increasing the exposure of segment s for asset a",
            records=segments[
                ["asset", "segment_id", "rel_origination_cost"]
            ].values.tolist(),
        )
        rel_sell_cost = gp.Parameter(
            domain=[A, S],
            description="Per unit cost of decreasing the exposure of segment s for asset a",
            records=segments[["asset", "segment_id", "rel_sell_cost"]].values.tolist(),
        )
        r = gp.Parameter(
            domain=[A, S],
            description="Risk weight of segment s for asset a",
            records=segments[["asset", "segment_id", "risk_weight"]].values.tolist(),
        )
        exposure = gp.Parameter(
            domain=[A, S],
            description="Current exposure of segment s for asset a.",
            records=segments[["asset", "segment_id", "exposure"]].values.tolist(),
        )
        current_asset_exposure = gp.Parameter(
            domain=A,
            description="Total exposure of asset a",
            records=segments.groupby("asset", as_index=False)["exposure"].sum(),
        )
        profit_stdev = gp.Parameter(
            domain=A, records=assets[["asset", "stdev_profitability"]].values.tolist()
        )
        correlation_df.set_index("asset", inplace=True)
        correlation_matrix = gp.Parameter(
            domain=[A, J], records=correlation_df, uels_on_axes=True
        )

        segment_vars = gp.Variable(
            domain=[A, S],
            description="Multiplier of original exposure for segment s, asset a in the rebalanced portfolio",
            type="Positive",
        )
        segment_increase_vars = gp.Variable(
            domain=[A, S],
            description="Increase multiplier of original exposure for segment s, asset a in the rebalanced portfolio",
            type="Positive",
        )
        segment_decrease_vars = gp.Variable(
            domain=[A, S],
            description="Decrease multiplier of original exposure for segment s, asset a in the rebalanced portfolio",
            type="Positive",
        )
        new_total_exposure = gp.Variable(
            description="Total exposure in new rebalanced portfolio",
            type="Positive",
            records={"level": options.new_total_exposure_initial_value},
        )
        portfolio_exposure_vars = gp.Variable(
            domain=A,
            description="Total exposure in new rebalanced portfolio for asset a",
            type="Positive",
        )
        portfolio_exposure_vars.lo[A] = (1 - lo_a[A]) * current_asset_exposure[A]
        portfolio_exposure_vars.up[A] = (1 + up_a[A]) * current_asset_exposure[A]

        ### --- Constraints --- ###
        # 1. Segment Relationship Constraint
        segment_relationship = gp.Equation(
            domain=[A, S],
            description="This establishes the relationship between the main segment variables and their increase/decrease components.",
        )
        segment_relationship[AS] = (
            segment_vars[AS]
            == 1 + segment_increase_vars[AS] - segment_decrease_vars[AS]
        )

        # 2. Asset Exposure Relationship Constraint
        asset_exposure = gp.Equation(
            domain=A,
            description="This establishes the relationship between the per segment ratio and the asset level exposure of the rebalanced portfolio.",
        )
        asset_exposure[A] = (
            gp.Sum(AS[A, S], exposure[AS] * segment_vars[AS])
            == portfolio_exposure_vars[A]
        )

        # 3. Risk Weight Constraint
        risk_weight = gp.Equation(
            description="This constraint keeps the average risk weight at the portfolio-level below the user-defined threshold.",
        )
        risk_weight[...] = (
            gp.Sum(AS, r[AS] * exposure[AS] * segment_vars[AS])
            <= risk_weight_limit * new_total_exposure
        )

        # 4. Total Exposure Relationship Constraint
        total_exposure = gp.Equation(
            description="This establishes the relationship between the per asset exposure and the total exposure of the rebalanced portfolio.",
        )
        total_exposure[...] = (
            gp.Sum(A, portfolio_exposure_vars[A]) == new_total_exposure
        )

        ### --- Objectives --- ###
        # 1. Maximize expected net profit
        profit = gp.Sum(AS, profitability[AS] * exposure[AS] * segment_vars[AS])
        transaction_cost = gp.Sum(
            AS,
            rel_origination_cost[AS] * exposure[AS] * segment_increase_vars[AS]
            + rel_sell_cost[AS] * exposure[AS] * segment_decrease_vars[AS],
        )
        profit_objective_variable = gp.Variable()
        net_profit = (
            profit - transaction_cost == profit_objective_variable
        )  ## Why is this done this way.
        profit_equation = gp.Equation(definition=net_profit)

        # 2. Minimize risk
        profit_downside_var = gp.Variable(type="Positive")
        variance = gp.Sum(
            (A, J),
            profit_stdev[A]
            * profit_stdev[J]
            * correlation_matrix[A, J]
            * portfolio_exposure_vars[A]
            * portfolio_exposure_vars[J]
            / (new_total_exposure * new_total_exposure),
        )
        risk_equation = gp.Equation(
            definition=z_score * profit * gp.math.sqrt(variance) <= profit_downside_var
        )

        alpha = gp.Parameter()
        model = gp.Model(
            problem=gp.Problem.QCP,
            sense=gp.Sense.MAX,
            equations=[
                risk_weight,
                segment_relationship,
                asset_exposure,
                total_exposure,
                profit_equation,
            ],
            objective=alpha * profit_objective_variable
            - (1 - alpha) * profit_downside_var,
        )

        if options.consider_risk:
            if options.profit_weight < 0:  # lexicographic
                alpha[...] = 1
                model.solve(solver="xpress", output=sys.stdout)
                rel_tol = 0.1
                profit_objective_variable.lo = profit_objective_variable.l * (
                    1 - rel_tol
                )

                alpha[...] = 0
                model.equations.append(risk_equation)
                model.problem = gp.Problem.NLP
                model.solve(
                    solver="xpress",
                    output=sys.stdout,
                    options=gp.Options(
                        relative_optimality_gap=options.relative_optimality_gap
                    ),
                )
            else:  # weighted
                model.equations.append(risk_equation)
                model.problem = gp.Problem.NLP
                alpha[...] = options.profit_weight
                model.solve(solver="xpress", output=sys.stdout)
        else:
            alpha[...] = 1
            model.solve(solver="xpress", output=sys.stdout)

        print(f"Net Profit: {profit_objective_variable.toValue()}")
        print(f"Expected Profit: {profit.toValue()}")
        print(f"Transaction Cost: {transaction_cost.toValue()}")
        print(f"Optimized Exposure: {new_total_exposure.toValue()}")
        print(f"Profit Downside: {profit_downside_var.toValue()}")

        output = build_output(
            options,
            segment_vars,
            exposure,
            portfolio_exposure_vars,
            current_asset_exposure,
            profit_objective_variable,
            profit,
            transaction_cost,
            new_total_exposure,
            profit_downside_var,
        )

        return output


def build_output(
    options: nextmv.Options,
    segment_vars: gp.Variable,
    exposure: gp.Parameter,
    portfolio_exposure_vars: gp.Variable,
    current_asset_exposure: gp.Parameter,
    profit_objective_variable: gp.Variable,
    profit: gp.Variable,
    transaction_cost: gp.Variable,
    new_total_exposure: gp.Variable,
    profit_downside_var: gp.Variable,
) -> nextmv.Output:
    # Extract solution data for segments
    segments_solution = []
    segment_vars_df = segment_vars.records
    exposure_df = exposure.records

    for _, row in segment_vars_df.iterrows():
        a = row["A"]  # Variable records use 'A'
        s = row["S"]  # Variable records use 'S'
        segment_multiplier = row["level"]

        # Find original exposure (parameters also use 'A' and 'S')
        exposure_row = exposure_df[(exposure_df["A"] == a) & (exposure_df["S"] == s)]
        if not exposure_row.empty:
            original_exposure = exposure_row["value"].values[0]
            segments_solution.append(
                {
                    "asset": a,
                    "segment_id": s,
                    "original_exposure": original_exposure,
                    "segment_multiplier": segment_multiplier,
                    "rebalanced_exposure": original_exposure * segment_multiplier,
                }
            )

    # Extract solution data for assets
    assets_solution = []
    portfolio_vars_df = portfolio_exposure_vars.records
    current_exposure_df = current_asset_exposure.records

    for _, row in portfolio_vars_df.iterrows():
        a = row["A"]  # Variable records use 'A'
        rebalanced_total = row["level"]

        # Find original exposure
        # Check column name - could be 'A' or 'asset'
        asset_col_name = "asset" if "asset" in current_exposure_df.columns else "A"
        original_row = current_exposure_df[current_exposure_df[asset_col_name] == a]
        if not original_row.empty:
            original_total = original_row["value"].values[0]
            assets_solution.append(
                {
                    "asset": a,
                    "original_exposure": original_total,
                    "rebalanced_exposure": rebalanced_total,
                    "exposure_change": rebalanced_total - original_total,
                }
            )

    stats = nextmv.Statistics(
        result=nextmv.ResultStatistics(
            custom={
                "net_profit": profit_objective_variable.toValue(),
                "expected_profit": profit.toValue(),
                "transaction_cost": transaction_cost.toValue(),
                "optimized_exposure": new_total_exposure.toValue(),
                "profit_downside": profit_downside_var.toValue(),
            },
        )
    )
    output = nextmv.Output(
        options=options,
        output_format=nextmv.OutputFormat.MULTI_FILE,
        statistics=stats,
        solution_files=[
            nextmv.csv_solution_file(name="segments", data=segments_solution),
            nextmv.csv_solution_file(name="assets", data=assets_solution),
        ],
    )

    return output


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segments = pd.read_csv(os.path.join("inputs", "segments.csv"))
    assets = pd.read_csv(os.path.join("inputs", "assets.csv"))
    correlation = pd.read_csv(os.path.join("inputs", "correlation.csv"))

    return segments, assets, correlation


if __name__ == "__main__":
    manifest = nextmv.Manifest.from_yaml(dirpath=".")
    options = manifest.extract_options()
    segments, assets, correlation = get_data()
    output = main(segments, assets, correlation, options)
    nextmv.write(output=output, path="outputs")

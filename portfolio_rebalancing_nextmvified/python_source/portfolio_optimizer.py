import nextmv
import xpress as xp
from domain import Portfolio


def optimize_portfolio(
        portfolio: Portfolio, 
        risk_weight_limit: float, 
        consider_risk: bool = True, 
        z_score: float = 1.96, 
        profit_weight: float=None,
        options: nextmv.Options=None,
    ) -> None:
    """
    Optimize the portfolio to maximize profit while keeping the average risk weight below a specified limit.
    
    Parameters:
    - portfolio: Portfolio object containing assets and segments information.
    - risk_weight_limit: Maximum allowable average risk weight for the entire portfolio.
    """

    print(f"Printing correlation matrix")
    print(portfolio.correlation_matrix)
    print(f"profit_weight {profit_weight} and consider_risk {consider_risk}")
    
    # Create a new optimization model
    model = xp.problem()
    
    # Decision variables: whether to increase origination (>1) or decrease for each segment (<1)
    segments = []
    for asset in portfolio.assets.values():
        for segment in asset.segments.values():
            segments.append((asset.asset_id, segment.segment_id))

    segment_vars = model.addVariables(segments, vartype=xp.continuous, lb=0.0)
    
    # Decision variable: Value of segment increase
    segment_increase_vars = model.addVariables(segments, vartype=xp.continuous, lb=0.0, name="increase")
    segment_decrease_vars = model.addVariables(segments, vartype=xp.continuous, lb=0.0, name="decrease")

    # Decision variable: Value of new total exposure
    new_total_exposure = model.addVariable(name="new_total_exposure", vartype=xp.continuous, lb=0.0)

    # Decision variable: Value of new exposure per portfolio
    portfolio_exposure_vars = model.addVariables(portfolio.assets.keys(), vartype=xp.continuous, lb=0.0)

    # Decision variable: RWA variance
    profit_downside_var = model.addVariable(name="RWA_variance", vartype=xp.continuous, lb=0.0)    

    # Constraint: Establish relationship between increase/decrease variables to segment variables i.e. x_s = 1 + x(increase)_s - x(decrease)_s for all s
    model.addConstraint(segment_vars[(asset.asset_id, segment.segment_id)] == 1 + segment_increase_vars[(asset.asset_id, segment.segment_id)] - segment_decrease_vars[(asset.asset_id, segment.segment_id)]
                        for asset in portfolio.assets.values()
                        for segment in asset.segments.values())

    # Constraint: Capture the new updated total exposure, i.e. sum_{s in S} exposure_s*x_s = new_total_exposure
    model.addConstraint(xp.Sum(segment.exposure * segment_vars[(asset.asset_id, segment.segment_id)]
                            for asset in portfolio.assets.values()
                            for segment in asset.segments.values()) == new_total_exposure)
    
    # Constraint: Keep average risk weight below the user-specified limit i.e. sum_{s in S} risk_s*exposure_s*x_s <= target_risk_weight * new_total_exposure
    model.addConstraint(xp.Sum(segment.exposure * segment.risk_weight * segment_vars[(asset.asset_id, segment.segment_id)]
                                        for asset in portfolio.assets.values()
                                        for segment in asset.segments.values()) <= risk_weight_limit * new_total_exposure)
    
    # Constraint: Capture asset exposures i.e. sum_{s in S_a} exposure_s*x_s = e_a
    model.addConstraint(xp.Sum(segment.exposure * segment_vars[(asset.asset_id, segment.segment_id)]
                            for segment in asset.segments.values()) == portfolio_exposure_vars[asset.asset_id] for asset in portfolio.assets.values())

    # Constraint: Keep portfolio exposures within allowable limits l_a<= e_a<= u_a
    model.addConstraint(portfolio_exposure_vars[asset.asset_id] >= asset.min_rel_exposure * asset.total_exposure for asset in portfolio.assets.values())
    model.addConstraint(portfolio_exposure_vars[asset.asset_id] <= asset.max_rel_exposure * asset.total_exposure for asset in portfolio.assets.values())

    # Objective: Maximize total profit
    profit = xp.Sum(segment.profitability * segment.exposure * segment_vars[(asset.asset_id, segment.segment_id)] for asset in portfolio.assets.values() for segment in asset.segments.values())
    transaction_cost = xp.Sum(segment.rel_origination_cost * segment.exposure * segment_increase_vars[(asset.asset_id, segment.segment_id)] 
                       +segment.rel_sell_cost * segment.exposure * segment_decrease_vars[(asset.asset_id, segment.segment_id)]
                       for asset in portfolio.assets.values()
                       for segment in asset.segments.values())
    net_profit_objective = profit-transaction_cost
    model.addObjective(net_profit_objective)

    # Considering portfolio variance in our optimization
    if consider_risk:
        # Constraint: Capture the variance based on covariance between assets
        variance = xp.Sum(asset_i.profit_stdev * asset_j.profit_stdev * 
                        portfolio.correlation_matrix.loc[asset_i.asset_id, asset_j.asset_id] * 
                        portfolio_exposure_vars[asset_i.asset_id] * portfolio_exposure_vars[asset_j.asset_id]/(new_total_exposure * new_total_exposure)
                        for asset_i in portfolio.assets.values()
                        for asset_j in portfolio.assets.values())
        model.addConstraint(z_score * profit * xp.sqrt(variance) <= profit_downside_var)
        # Insert the objective function
        variance_objective = profit_downside_var
        model.addObjective(variance_objective)

        # Combine the objectives
        if profit_weight >= 0:
        # Set the objectives for weighted approach
            model.setObjective(objidx=0, weight=profit_weight, sense=xp.maximize)
            model.setObjective(objidx=1, weight=profit_weight-1)
        else:
            # Set the objectives in lexicographic order: first maximize profit, then minimize variance
            model.setObjective(objidx=0, priority=1, weight=1, reltol=0.1,sense=xp.maximize)
            model.setObjective(objidx=1, priority=0, weight=-1)
    else:
        model.setObjective(objidx=0, priority=1, weight=1, sense=xp.maximize)
    
    # Control settings
    model.controls.outputlog = options.outputlog
    model.controls.bariterlimit = options.bariterlimit
    model.controls.miprelstop = options.miprelstop
    model.setOutputEnabled(options.setoutputenabled)
    
    # Solve the optimization problem
    _, solstatus = model.optimize()

    # Solution processing
    investment = {}
    expected_profit=0
    profit_stdev=0
    new_exposure = 0
    if solstatus in (xp.SolStatus.OPTIMAL,xp.SolStatus.FEASIBLE):
        investment = model.getSolution(segment_vars)
        expected_profit = xp.evaluate(net_profit_objective, problem=model)
        profit_stdev = model.getSolution(profit_downside_var)/(z_score*xp.evaluate(profit, problem=model))
        new_exposure = model.getSolution(new_total_exposure)
        print(f"Optimized Segment Decisions achieving total expected net profit of ${expected_profit} and Portfolio "
        f"Standard deviation of {profit_stdev},  {new_exposure} from initial exposure of {portfolio.total_exposure}.")

    else:
        print(f"Unable to find solution solve status of {solstatus}.")
    
    return investment

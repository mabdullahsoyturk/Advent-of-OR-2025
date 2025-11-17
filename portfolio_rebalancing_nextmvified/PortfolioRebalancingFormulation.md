# Portfolio Rebalancing Model Documentation

## Overview

This document describes the mathematical formulation, notation, parameters, and variables used in the Xpress optimization model implemented in Python for the Portfolio Rebalancing Optimization application.

The goal of this optimization problem is to optimally rebalance a portfolio comprised of assets (mortgage, revolving, personal loans, etc.) and different segments of the assets (e.g. prime, subprime, etc.) considering:

- profitability of each segment
- profitability variance for each asset for multiple scenarios
- correlation of profitability for each asset for multiple scenarios
- transaction costs for rebalancing
- risk weights for each segment 
- average risk weight limit at the portfolio level
- bounds on the maximum and minimum exposure of each asset in the rebalanced portfolio


## Mathematical Formulation
### Sets and Indices

| Set | Description |
|-----|-------------|
| $A$ | Set of all assets in the portfolio |
| $S_a$ | Set of segments for asset `a` |
| $\Omega$ | Set of profit variability scenarios |
| $(a,s)$ | Asset-segment pairs for asset `a` and segment`s` |

### Parameters

| Parameter | Description |
|-----------|-------------|
| $P_{as}$ | Expected profitability for asset `a`, segment `s` |
| $\Gamma$ | Maximum allowable average risk weight at the portfolio level |
| $z$ | Z-score for risk calculation (default: 1.96 corresponding to 95% confidence level) |
| $l_a$, $u_a$ | Lower and upper bounds for asset `a` relative exposure |
| $\hat{c}_{as}$, $\bar{c}_{as}$ | Per unit cost of increasing and decreasing, respectively, the exposure of segment `s` for asset `a` |
| $r_{as}$ |  Risk weight of segment `s` for asset `a` |
| $E_{as}$ |  Current exposure of segment `s` for asset `a` |
| $E_{a}$ |  Current exposure of  asset `a` |
| $\sigma^{\omega}_{a}$ |  Profit Standard Deviation of asset `a` in scenario $\omega$|
| $\rho^{\omega}_{ij}$ |  Correlation between assets `i` and `j`in scenario $\omega$|

### Decision Variables

| Variable |  Description |
|----------|-------------|
| $x_{as}$ |  Multiplier of original exposure for segment `s`, asset `a` in the rebalanced portfolio  |
| $\hat{x}_{as}$ | Increase multiplier of original exposure for segment `s`, asset `a` in the rebalanced portfolio  |
| $\bar{x}_{as}$ | Decrease multiplier of original exposure for segment `s`, asset `a` in the rebalanced portfolio  |
| $\hat{E}$ |  Total  exposure in new rebalanced portfolio |
| $\hat{e}_a$ | Total exposure in new rebalanced portfolio for asset `a`  |
| $\gamma$ | Profit variance |


Based on the code structure, the optimization model can be formulated as follows:

### Objective Functions

**Maximize: Expected Net Profit = Expected Profit - Transaction Costs**

$\text{maximize } \sum_{a \in A, s \in S} \left(P_{as}x_{as} - \hat{c}_{as}\hat{x}_{as} - \bar{c}_{as}\bar{x}_{as} \right)E_{as}$

Note: 

- We distinguish between the expected profit which is the ratio of profit per \$ of exposure from net profit which accounts for both expected profit and the transaction costs to rebalance.
- Because the paramaters and variables represent ratios, we need to multiply by the original exposure of each to determine the nominal values

**Minimize: Profit Variability of all Ordered Scenarios**

To ensure the rebalancing strategy is robust to different volatility scenarios that can come about from macro economic, policy, and even trade changes, our approach will be adoptable to a user providing N scenarios for profit variability, each with its priority. The priority will dictate how important that scenario is in terms of optimization. For example, a scenario with priority 1, should be the scenario with highest likelihood and for which we want our optimization to be most optimized on, while a scenario with priority 5 should be a scenario with least likelihood of ocurring.

To simplify notation, we will use $\omega$ to denote a specific profit variability scenario in the set of profit variability scenarios the user defines.

$\text{minimize } \gamma_{\omega}$

where 
$\gamma_{\omega} \geq z \left(\sum_{a \in A, s \in S}P_{as}E_{as}x_{as}\right)\sqrt{\sum_{i \in A, j \in A}\sigma^{\omega}_{i}\sigma^{\omega}_{j}\rho^{\omega}_{ij}\left(\frac{\hat{e}_{i}}{\hat{E}}\right)\left(\frac{\hat{e}_{j}}{\hat{E}}\right)}$

is imposed as a constraint.

After solving each scenario of profit variability, the following optimization problem will then include the constraint for the new scenario and an additional constraint to the objective function of the previous scenario to ensure it does not deviate too far from optimal.

Note:

- $\gamma$ is measured in dollars \$. The logic behind $\gamma$ is that it captures the potential downside due to profit variability in each asset and amongst the assets themselves (correlation).
- the term $z$ represents the z-score which is the critical value of a normal distribution for a given confidence interval which can be modified by the user through the UI. We assume a default confidence interval of 95% which translates to a z-score of approximately 1.964. 
- the right hand side of the equation leverages the same concept as safety stocks in inventory theory where, given a confidence interval we hedge over the variability of the performance of a random variable. In this case the random variable is the expected profit of the rebalanced portfolio.


### Key Constraints
The following constraints impose limits to the set of actions that can be realized. These can be controlled by the user through the UI parameters.

#### 1. Asset Exposure Bounds

This constraint keeps asset exposures within allowable limits.

$l_aE_a \leq \hat{e}_a \leq u_a E_a  \qquad \forall a \in A$


#### 2. Risk Weight Constraint

This constraint keeps the average risk weight at the portfolio-level below the user-defined threshold.

$\sum_{a \in A, s\in S}r_{as}E_{as}x_{as} \leq \Gamma \hat{E}$

### Complementary Constraints
The following constraints serve to establish the link among the different sets of variables used in the model.

#### 1. Segment Relationship Constraint

This establishes the relationship between the main segment variables and their increase/decrease components.

$x_{as} = 1 + \hat{x}_{as} - \bar{x}_{as} \qquad \forall a \in A, s \in S_a$


#### 2. Asset Exposure Relationship Constraint

This establishes the relationship between the per segment ratio and the asset level exposure of the rebalanced portfolio.

$\sum_{s\in S_a}E_{as}x_{as} = \hat{e}_a  \qquad \forall a \in A$



#### 3. Total Exposure Relationship Constraint

This establishes the relationship between the per asset exposure and the total exposure of the rebalanced portfolio.

$\sum_{a \in A} \hat{e}_a = \hat{E}$


## Solution Interpretation

The optimal solution provides:

- **Investment Strategy**: Values of $x_{as}$ indicating how to scale each segment
- **Expected Profit**: Total expected return from the optimized portfolio
- **Transaction Costs**: Total Transaction costs
- **Profit Variability**: The potential profit by which our profit may be overestimated due to variability and correlation between assets' profit.
Align trade strategy with the expected market conditions based on historical patterns and observed volatility dynamics.

# Vol Parameters and option trade Strategies:
Implied Volatility (IV):
Increase in IV: Straddle and Strangle, benefit from larger price movements in either direction.
Decrease in IV: Butterfly and Iron Condor, profit from stable or decreasing volatility

Realized Volatility (RV):
Higher than IV: Long Straddles or Strangles
Lower than IV: Short Iron Condors

IV Term Structure:
Steepening (ST IV > LT IV): Calendar Spreads: Short short-term and long long-term options
Flattening (ST IV < LT IV): Reverse Calendar Spreads: Long short-term and short long-term options.

Skew (Up Skew / Down Skew):
Increased Up Skew: Long Call
Increased Down Skew: Long Put

Option Convexity:
High Convexity (significant curvature): Long Straddles or Long Strangles: benefit from large price movements.
Low Convexity (less curvature): Short Iron Condors: where small price movements are favorable.

# Expectations
<img width="624" alt="Screenshot 2024-01-03 at 00 14 34" src="https://github.com/Aurelie-Dubost/xeu/assets/61312165/a81a01c6-a542-45cd-9484-e27949cd2cd3">

# Modelling
Setting thresholds for each parameter. For instance:
- If IV increases by X%: Opt for a Straddle/Strangle.
- If RV is consistently higher than IV by Y%: Consider Long Volatility strategies.
- If Short-term IV exceeds Long-term IV by Z%: Look into Calendar Spreads.

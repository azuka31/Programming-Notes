# A Crash Course in Causality
# Confusion over Causality
# Potential Outcomes and Counterfactuals
# Hypothetical interventions
# Causal Effects
# Causal Assumptions
Identifiability: requires some untestable assumptions
The most common causal assumptions:
- Stable Unit Treatment Value Assumption (SUTVA)
  1. No interference:
  - Units do not interfere with each other
  - Doesn't affect that outcome of another unit
  - Spillover or contagion
  2. One version of treatment
- Consistency Assumption
  
  Potential outcome Y<sup>a</sup> = observed outcome Y|A=a
  
  > E(Y|A=a, X=x) = E(YY<sup>a</sup>|A=a, X=x)
- Ingorability Assumption

  Treatment assignment is independent from potential outcomes
  
  > Y<sup>0</sup>, Y<sup>1</sup> || A|X 
  > 
  > E(Y|A=a, X=x) = E(Y|X=x)
  
  (treatment A being randomly assigned, A and X covariates)
  
- Positivity Assumption
  
  Every X value is not zero probability, treatment was not deterministic.
  
  > P(A=a|X=x)>0
  
### Observed Data and Potential Outcomes

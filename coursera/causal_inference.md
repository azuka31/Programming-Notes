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
  
  > E(Y|A=a, X=x) = E(Y<sup>a</sup>|A=a, X=x)
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
# Stratification
Conditioning and marginalizing

E(Y<sup>a</sup> = SIGMA E(Y|A=a, X=x)P(X=x)

 X | Mace Y| Mace N|Total
---|---|---|---
saxa Y|350|3650|4000
saxa N|500|6500|7000
Total|750|10250|11000

Probabitiltiy Mace Y, Saxa Y = 350/4000 = 0.088
Probabitiltiy Mace Y, Saxa N = 500/7000 = 0.071

> Saxa Higher risk

Standarization Example

**Table of Prior OAD = N**
 X | Mace Y| Mace N|Total
---|---|---|---
saxa Y|50|950|1000
saxa N|200|3800|4000
Total|250|4750|5000

P of Mace, Saxa = Y = 5%

P of Mace, Saxa = N = 5%

**Table of Prior OAD = Y**
 X | Mace Y| Mace N|Total
---|---|---|---
saxa Y|300|2700|3000
saxa N|300|2700|3000
Total|600|5400|6000

P of Mace, Saxa = Y = 10%

P of Mace, Saxa = N = 10%

E(Y<sup>saxa</sup>) = E(Y|A=saxa, X=OAD)P(OAD) + E(Y|A=saxa, NOAD)P(NOAD)

= (0.1)(6000/11000) + (5%)(5000/11000) = 7.7%

# Incident user and active comparator designs

Cross-sectional look at treatments

![ce1](output/causal_effects1.png)

Incident users - need treatment known to identify how they start
Active Competitos - needs comparative alike activiti, like zumba and yoga

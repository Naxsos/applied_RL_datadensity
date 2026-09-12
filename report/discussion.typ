= Discussion Gerrit: Pendulum 
#v(-10pt)
\
- Interpretation: why learning $alpha$ outperforms a fixed $p$ -- it adapts to how much penalty
  is actually needed to satisfy the constraint, rather than requiring a priori knowledge.
- Where the Lagrangian underperforms or adds complexity: extra hyperparameters ($eta$,
  $epsilon$, $alpha_"init"$), sensitivity to update frequency, transient behavior early in
  training.
- What the LunarLander result adds or complicates: confirms generalization, or surfaces an
  environment-specific caveat.
- Limitations: single cost signal (KDE) tested; constraint definition (zone rate vs. mean cost)
  changes what $epsilon$ means; seed count; compute budget.
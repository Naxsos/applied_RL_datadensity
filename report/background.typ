= Background / Problem Formulation Theresa

- Mention previous paper from other group and their contribution 

- Penalized reward formulation:

$ "effective reward" = R(s, a) - w(c) dot c(s, a) $ <eq:penalized-reward>

  where $R(s,a)$ is the clean task reward, $c(s,a)$ is a data-density cost signal, and $w(c)$
  is the penalty weight.
- Cost signal used: KDE density estimate, held fixed across both conditions so the comparison
  isolates the weighting axis only.
- Two weighting schemes under test:
  - Fixed weight: $w(c) = p$, a constant (paper baseline).
  - Lagrangian weight: $w(c) = alpha$, updated online via
    $ alpha <- max(0, alpha + eta (C - epsilon)) $ <eq:dual-ascent>
    where $C$ is the observed constraint value (violation rate or mean cost) and $epsilon$ is
    the target.
- Why this isolates a clean research question: transition model, agent, evaluation loop, and
  seeds are identical between conditions -- only *how much* to penalize differs.
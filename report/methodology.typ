= Methodology

#let optional-internal-link(target, body) = context {
  if query(target).len() > 0 {
    link(target, body)
  } else {
    body
  }
}

== Environments: Jonas: Lunar Lander
- *Primary: Pendulum swing-up with excluded zone.* 
The Pendulum experiments are based on the continuous-control `Pendulum-v1`
environment. The observation is
$s_t = (cos theta_t, sin theta_t, dot(theta)_t)$, and the action is a continuous
torque $u_t in [-2, 2]$. The objective is to bring the pendulum to the upright
position ($theta = 0$) starting from the default `Pendulum-v1` reset distribution.

Safety is represented by an excluded angular interval defined on the signed
angle, i.e., the interval is one-sided and is not mirrored using $abs(theta)$.
Environment E1 uses

$
  theta in [2 pi / 5, 3 pi / 5]
  = [72 degree, 108 degree],
$

while E3 uses the wider interval

$
  theta in [pi / 4, 3 pi / 4]
  = [45 degree, 135 degree].
$

E1 and E3 share the same Pendulum dynamics and differ only in the geometry of
the excluded region. E3 was introduced to test whether a penalty configuration
that works for E1 remains effective under a more restrictive safety region.

A reference-policy analysis confirmed a
zone-free swing-up for E1. For the current E3 definition, however, no zone-free
path was found, so its feasibility for a fair safety comparison has not yet
been established.

- *Generalization: LunarLander.* #emph[TODO: define the excluded-zone / low-data-region
  analogue for this environment, and why it is expected to stress-test the same claim.]
- Shared components held fixed across both methods per environment: transition model /
  simulator, SAC hyperparameters, training steps, seeds.

== Methods compared

*Baseline.*
The baseline uses a KDE-based density cost to penalize states that are poorly
represented in the offline dataset. The penalty weight $p$ is fixed during
training and is evaluated over a predefined parameter sweep:

$
  r_"train" = r_"task" - p dot c(s_t).
$

*Lagrangian method (`lagr`).*
The Lagrangian method uses the same KDE-based cost as the baseline, but replaces
the fixed penalty weight $p$ with a multiplier $alpha$ that is adapted during
training. Instead of manually choosing how strongly low-density regions should
be penalized, the method defines a maximum tolerated constraint value $epsilon$:

$
  C(pi) <= epsilon,
$

where $C(pi)$ denotes the expected accumulated KDE cost under policy $pi$.
The constrained optimization problem can be written as

$
  max_pi J(pi)
  quad "subject to" quad
  C(pi) <= epsilon.
$

Using the Lagrangian formulation, the training objective becomes

$
  L(pi, alpha)
  =
  J(pi) - alpha (C(pi) - epsilon),
$

with $alpha >= 0$. During training, the multiplier is updated according to the
observed constraint violation:

$
  alpha <- max(0, alpha + eta_alpha (C(pi) - epsilon)).
$

If the constraint is violated, $alpha$ increases and the agent is penalized
more strongly for entering low-density regions. If the constraint is satisfied,
$alpha$ can decrease. The method can therefore adapt the penalty strength
automatically instead of relying on a fixed manually selected value of $p$.

*Uncertainty-based methods.*

The ensemble method (`ens`) trains multiple transition models and uses the
disagreement between their next-state predictions as an estimate of epistemic
uncertainty. States with high model disagreement therefore receive a larger
penalty.

The Bayesian neural network method (`bnn`) follows the same idea, but estimates
uncertainty using a single Bayesian transition model instead of several
independently trained models.

All four methods were evaluated in an initial exploratory experiment
(see Appendix: #optional-internal-link(<appendix-pilot-experiment>, [Initial exploratory experiment])).
The ensemble and BNN methods showed weaker preliminary performance than the
Lagrangian approach. Based on these initial findings, the main comparison was
therefore restricted to the Lagrangian method and the fixed-weight baseline.


== Training and evaluation protocol Maram 
- Offline dataset generation per environment.
- SAC training, $N$ steps, $>=5$ seeds per cell, mean $plus.minus$ std reported.
- Evaluation always on the clean, unpenalized reward, so methods with different training
  objectives remain comparable.
- Fixed start state, fixed episode length, greedy (deterministic) action evaluation.

== Metrics Theresa <sec-metrics>
- *Safety (lower is better):* zone visit rate (episode-level), zone step fraction, left/right
  path split (route-around behavior), zone depth-weighted step fraction (mean per-step
  penetration depth, geometric distance from the nearest zone edge normalized to
  $[0, 1]$ with $0$ outside the zone and $1$ at its center; folds frequency and
  severity into one number so a policy that only clips the boundary doesn't score the
  same as one that crosses through the center -- unlike the flat zone step fraction,
  which the KDE cost signal's "bleeding" near the zone edge can make misleadingly
  similar across methods that differ a lot in how deep they actually go).
- *Task performance (higher is better):* true return mean/std, upright success rate,
  time-to-upright.
- *Constraint behavior (Lagrangian only):* final $alpha$, $alpha$ trajectory over training,
  constraint value $C$ vs. $epsilon$, constraint-satisfied flag.
- *Robustness / tunability:* spread of (return, zone rate) across each method's own knob sweep;
  a flat spread means the method is easy to tune -- the Lagrangian's expected advantage.
- *Compute:* wall-clock training time, where relevant to a claim.
- For LunarLander: same categories, redefined for that environment's task-success and
  "excluded zone" analogue.

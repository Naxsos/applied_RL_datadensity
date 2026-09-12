= Methodology

== Environments: Jonas: Lunar Lander
#v(-10pt)
\
*Primary: Pendulum swing-up with excluded zone.* 
The Pendulum experiments are based on the continuous-control `Pendulum-v1`
environment. The observation is
$s_t = (cos theta_t, sin theta_t, dot(theta)_t)$, and the action is a continuous
torque $u_t in [-2, 2]$. The objective is to bring the pendulum to the upright
position ($theta = 0$) starting from the default `Pendulum-v1` reset distribution.
#v(-10pt)
\
Safety is represented by an excluded angular interval defined on the signed angle, i.e., the interval is one-sided and is not mirrored using $abs(theta)$.
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
#v(-10pt)
\
E1 and E3 share the same Pendulum dynamics and differ only in the geometry of
the excluded region. E3 was introduced to test whether a penalty configuration
that works for E1 remains effective under a more restrictive safety region.
#v(-10pt)
\
A reference-policy analysis confirmed a
zone-free swing-up for E1. For the current E3 definition, however, no zone-free
path was found, so its feasibility for a fair safety comparison has not yet
been established.
#v(-10pt)
\
- *Generalization: LunarLander.* #emph[TODO: define the excluded-zone / low-data-region
  analogue for this environment, and why it is expected to stress-test the same claim.]
- Shared components held fixed across both methods per environment: transition model /
  simulator, SAC hyperparameters, training steps, seeds.

== Methods compared
#v(-10pt)
\
*Baseline.*
The baseline uses a KDE-based density cost to penalize states that are poorly
represented in the offline dataset. The penalty weight $p$ is fixed during
training and is evaluated over a predefined parameter sweep:

$
  r_"train" = r_"task" - p dot c(s_t).
$
#v(-10pt)
\
*Lagrangian method (`lagr`).*
The Lagrangian method uses the same KDE-based cost as the baseline, but replaces
the fixed penalty weight $p$ with a multiplier $alpha$ that is adapted during
training. Instead of manually choosing how strongly low-density regions should
be penalized, the method defines a maximum tolerated constraint value $epsilon$:

$
  C(pi) <= epsilon,
$
#v(-10pt)
\
where $C(pi)$ denotes the expected accumulated KDE cost under policy $pi$.
The constrained optimization problem can be written as

$
  max_pi J(pi)
  quad "subject to" quad
  C(pi) <= epsilon.
$
#v(-10pt)
\
Using the Lagrangian formulation, the training objective becomes

$
  L(pi, alpha)
  =
  J(pi) - alpha (C(pi) - epsilon),
$
#v(-10pt)
\
with $alpha >= 0$. During training, the multiplier is updated according to the
observed constraint violation:

$
  alpha <- max(0, alpha + eta_alpha (C(pi) - epsilon)).
$
#v(-10pt)
\
If the constraint is violated, $alpha$ increases and the agent is penalized
more strongly for entering low-density regions. If the constraint is satisfied,
$alpha$ can decrease. The method can therefore adapt the penalty strength
automatically instead of relying on a fixed manually selected value of $p$.
#v(-10pt)
\
*Uncertainty-based methods.*
#v(-10pt)
\
The ensemble method (`ens`) trains multiple transition models and uses the
disagreement between their next-state predictions as an estimate of epistemic
uncertainty. States with high model disagreement therefore receive a larger
penalty.
#v(-10pt)
\
The Bayesian neural network method (`bnn`) follows the same idea, but estimates
uncertainty using a single Bayesian transition model instead of several
independently trained models.
#v(-10pt)
\
All four methods were evaluated in an initial exploratory experiment
(see Appendix: #(<appendix-pilot-experiment>, [Initial exploratory experiment])).
The ensemble and BNN methods showed weaker preliminary performance than the
Lagrangian approach. Based on these initial findings, the main comparison was
therefore restricted to the Lagrangian method and the fixed-weight baseline.


== Training and evaluation protocol Maram <sec-protocol>
- Offline dataset generation per environment.
- SAC training, $N$ steps, $>=5$ seeds per cell, mean $plus.minus$ std reported.
- Evaluation always on the clean, unpenalized reward, so methods with different training
  objectives remain comparable.
- Fixed start state, fixed episode length, greedy (deterministic) action evaluation.

== Metrics <sec-metrics>
#v(-10pt)
\
All metrics are computed from deterministic evaluation episodes in the clean, unpenalized
environment. Episode-level quantities are calculated for each episode and then averaged within
a run; the reported $plus.minus$ values aggregate the corresponding run-level means across
seeds. The episode length is the number of transitions actually executed, so terminated and
truncated episodes do not contribute additional padded steps.
#v(-10pt)
\
*Safety and route diagnostics (lower is better where applicable).*
- _Zone visit rate_: the fraction of evaluation episodes that contain at least one observed
  state in the excluded zone. This is an episode-level measure; a single-step boundary crossing
  counts as a visit.
- _Zone step fraction_: for episode $e$, $T_e^(-1) sum_(t=1)^(T_e) z_(e,t)$, where
  $z_(e,t) = bb(1)[s_(e,t) in Z]$. It is the average fraction of executed steps spent in the
  zone and is distinct from the episode-level visit rate used above.
- _Zone depth-weighted step fraction_: the episode mean of an environment-specific depth
  function $d(s) in [0, 1]$. For the pendulum, $d$ is zero outside the angular band and at
  its edges, and increases linearly to one at the band centre. For a LunarLander box zone,
  it is the minimum of the normalized distances to the two pairs of box boundaries; for the
  legacy corridor zone, it reduces to the binary zone indicator. Averaging $d(s)$ combines
  entry frequency with penetration severity, so shallow boundary contacts contribute less than
  trajectories through the zone interior.
- _Left/right path split_ (pendulum only): the sign of the cumulative unwrapped angle change
  determines the code's route label: positive travel is recorded as ``right'' and non-positive
  travel as ``left''. The two percentages are descriptive route diagnostics, not additional
  constraint values, and are undefined for LunarLander.
#v(-10pt)
\
*Task performance (higher is better unless stated otherwise).*
- _True return_: the undiscounted sum of clean task rewards over an episode. The evaluator
  stores the episode mean and standard deviation for each run; tables compare the means across
  seeds.
- _Upright success rate_ (pendulum): the fraction of episodes reaching $abs(theta) < 0.2$
  radians at any evaluated step.
- _Time to upright_ (pendulum): the first evaluated step satisfying the upright threshold;
  episodes that never reach it are assigned the episode limit (200 steps in E1/E3).
- _Strict landing rate_ (LunarLander): the fraction of episodes ending in a terminal event
  with a positive terminal reward, corresponding to the environment's $+100$ landing bonus.
- _Landing success rate_ (LunarLander): the strict landing events together with episodes whose
  final state is touchdown-like (near the pad, sufficiently stable, or showing leg contact),
  including timeouts. It is therefore a more permissive task-success measure.
- _Crash rate_ and _timeout rate_ (LunarLander): terminal events with a non-positive terminal
  reward and episodes ending at the 500-step limit without termination, respectively. These
  categories are mutually exclusive with strict landing and partition the evaluated episodes.
- _Mean episode length_: the average number of executed transitions.
#v(-10pt)
\
*Constraint diagnostics (Lagrangian only).* The implementation logs the final multiplier
$alpha$, its trajectory over dual updates, the constraint value $C$, the target $epsilon$, and
whether the final update satisfies $C <= epsilon$. When the Lagrangian uses the ``zone''
constraint, $C$ is the fraction of *training steps* inside the excluded zone; when it uses the
``cost'' constraint, $C$ is the mean normalized cost signal. Neither quantity is the
per-episode zone visit rate reported at evaluation.
#v(-10pt)
\
*Robustness and compute.* For each method, robustness is assessed from the spread of clean
return and zone metrics across its own tuning-parameter sweep ($p$ for the baseline and
$epsilon$ for the Lagrangian). Wall-clock training time is recorded with evaluation excluded;
the Lagrangian's additional operation is the scalar dual update performed at the configured
interval.

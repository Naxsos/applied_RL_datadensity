= Methodology

#let optional-internal-link(target, body) = context {
  if query(target).len() > 0 {
    link(target, body)
  } else {
    body
  }
}

== Environments
#v(-10pt)
\

*Primary: Pendulum swing-up with excluded zone.*
The Pendulum experiments are based on the continuous-control `Pendulum-v1`
environment. The observation is
$s_t = (cos theta_t, sin theta_t, dot(theta)_t)$, and the action is a continuous
torque $u_t in [-2, 2]$. The objective is to bring the pendulum to the upright position
($theta = 0$), starting from a resting configuration near
($theta = pi$ and $dot(theta) = 0$), with small random perturbations
of magnitude $0.05$ applied to the initial state.

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

*Generalization: LunarLander.*
The LunarLander generalization task uses the continuous-control `LunarLander-v3` environment,
a standard Gymnasium benchmark for contact-rich control problems in which the agent must
land a craft on a pad while preserving stability @towers2024gymnasium.
The observation is $s_t = (x_t, y_t, dot(x)_t, dot(y)_t, theta_t, dot(theta)_t, l_t, r_t)$, where
$(x, y)$ denotes horizontal and vertical position, $dot(x), dot(y)$ are velocities,
$theta$ is attitude angle, $dot(theta)$ is angular velocity, and $l, r in {0, 1}$ indicate left and right
leg ground contact. The action is a discrete selection from four thrusters (no action, main engine, left-only,
right-only). The objective is to land smoothly at the target position $(0, 0)$ @gymnasium_lunar_lander.
#v(-10pt)
\
As illustrated in the offline-data figure (@fig-ll-offline-zone), an excluded box-shaped
region in $(x, y)$ position space is constructed as an artificial low-density zone rather than
as a literal no-landing restriction in the environment. All offline trajectories that enter
this box were removed from the training data, so the region becomes a sparse-coverage area
with effectively zero observed samples:

$
  (x, y) in [-0.2, 0.2] times [0.6, 1.0].
$
#v(-10pt)
\
This box is therefore not a hard airspace exclusion in the simulator; it is a dataset-design
feature that induces a low-data region in the offline dataset. The region remains navigable by
steering left or right, analogous to the pendulum's excluded angular interval. Unlike the
pendulum's one-dimensional angular constraint, the box zone allows richer navigation
strategies while remaining a deliberately under-covered area in the offline data. The purpose
is to test whether the density-aware methods continue to avoid the sparse region when the task
requires passing near it en route to the landing pad, mirroring the sparse-coverage setups
used in prior density-penalized offline RL work @lantz2025.

#figure(
  image("../figures/ll_offline_with_zone.png", width: 100%),
  caption: [LunarLander offline trajectories with excluded box-zone visualization.
  The red rectangle marks the excluded region in position space where the agent should avoid lingering.],
) <fig-ll-offline-zone>

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
(see Appendix: #optional-internal-link(
  <appendix-pilot-experiment>,
  [Initial exploratory experiment],
)).
The ensemble and BNN methods showed weaker preliminary performance than the
Lagrangian approach. Based on these initial findings, the main comparison was
therefore restricted to the Lagrangian method and the fixed-weight baseline.


== Training and evaluation protocol <sec-protocol>
*Offline dataset generation.*
Prior to policy optimization, a fixed offline dataset was generated separately
for each environment. Data were collected using random actions sampled from the
corresponding action space over 2,000 episodes, with a maximum of 200 steps per
episode. Environment resets and action sampling were seeded for reproducibility.

Each interaction produced a transition $(s_t, a_t, s_(t+1))$. If the next state
$s_(t+1)$ lay inside the environment-specific excluded region, the transition
was not stored, while the episode continued normally unless the environment
terminated. This created a low-density region in the offline data without
modifying the underlying environment dynamics.

The collected transitions were stored as Parquet files containing the current
observation, action, and next observation. The datasets were not used to train
the policy directly. Instead, they were used to fit the KDE-based safety signal
before policy optimization. The fitted KDE was then kept fixed during training,
assigning higher safety costs to states with low estimated data density.

*Policy training and configurations.*
For each task, the comparison consists of three fixed-weight baseline
configurations and one adaptive Lagrangian configuration. The baseline uses
$p in {2, 10, 30}$, with the selected penalty weight remaining constant
throughout training. These three configurations represent different manually
chosen levels of safety penalization.

The Lagrangian method replaces the manually selected value of $p$ with a
non-negative multiplier $alpha$. The multiplier is initialized at
$alpha_0 = 5$ and updated every 500 environment steps using a dual learning
rate of $eta_alpha = 10$. Its update depends on the difference between the
observed mean safety cost and the allowed threshold $epsilon = 0.01$. When the
constraint is violated, $alpha$ increases and the safety penalty becomes
stronger. When the constraint is satisfied, $alpha$ can decrease. The method
therefore adapts the penalty strength continuously instead of selecting one of
the three predefined baseline weights.

All policies are trained using the Stable-Baselines3 implementation of Soft
Actor-Critic (SAC) with an MLP policy for 150,000 environment steps. The KDE
safety signal uses a bandwidth of 0.1 and a binary density threshold of 0.025.
A state receives a safety cost of 1 when its estimated density is below this
threshold and a cost of 0 otherwise.

Each of the four configurations is repeated with six independent random seeds,
numbered 0 through 5. This produces 18 fixed-weight baseline runs and six
Lagrangian runs, for a total of 24 runs per task. NumPy, PyTorch, and SAC use
the same seed within each run. Within a task, all methods use the same offline
dataset and environment construction.

*Evaluation procedure.*
During training, the original task reward is modified by the weighted KDE
safety cost. Intermediate evaluations are performed every 10,000 training
steps using 50 episodes. These evaluations monitor policy development without
affecting the policy updates.

Each policy is evaluated over 200 episodes, with
a maximum length of 200 steps per episode. Evaluation is performed in a
separate, unpenalized environment using deterministic actions. Returns are
calculated exclusively from the original task reward, while safety is measured
separately using the zone-visit rate and the fraction of steps spent inside the
excluded region. The final results are aggregated across the six seeds and
reported using the mean and standard deviation.

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

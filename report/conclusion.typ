= Conclusion <sec-conclusion>
#v(-10pt)
\
This report compared a fixed density-penalty coefficient $p$ @lantz2025 with a Lagrange
multiplier learned online from a user-specified tolerance $epsilon$. Under the shared
experimental design, adaptive weighting improved the safety-performance trade-off on
Pendulum and, on LunarLander, was competitive with the fixed-weight sweep on task performance
with fewer crashes, using the same dual setting without retuning. It replaces the fixed-weight sweep
with a tolerance specification, while retaining dual hyperparameters that require calibration.
#v(-10pt)
\
== Pendulum.
#v(-10pt)
\
On the paper's task with its one-sided excluded band, one Lagrangian setting,
$epsilon = 0.01$, matched or exceeded the pooled fixed-weight sweep $p in {2, 10, 30}$ across
every metric of @tab-pendulum-pooled: depth-weighted zone occupancy lower by a factor of
$4.2$ ($0.075 plus.minus 0.077%$ of steps compared with $0.316 plus.minus 0.359%$), a marginally
better clean return ($-331.2 plus.minus 9.5$ compared with $-337.1 plus.minus 7.1$), upright success
in every episode of every seed ($100%$ compared with $94.4 plus.minus 22.9%$), a faster and
substantially more consistent swing-up ($54.2 plus.minus 4.1$ compared with $64.3 plus.minus 33.4$ steps), and
the same wall-clock cost ($789$ s compared with $787$ s). All six seeds followed a trajectory around the band.
#v(-10pt)
\
The individual baseline settings show the trade-off:
$p = 2$ penalizes too weakly, at $0.76%$ depth-weighted occupancy and with a swing direction
that changes from seed to seed; $p = 10$ approaches the Lagrangian on return but enters the
band roughly $2.5 times$ as deeply; and $p = 30$ removes band entries in five seeds out of six
while being the only configuration to fail the task entirely for one seed, never reaching an
upright state in 200 episodes.
#v(-10pt)
\
The multiplier trajectory in @fig-alpha-trajectory shows the mechanism: the mean of $alpha$ climbs from $5$ to roughly $12.5$ for as
long as the early policy violates the constraint, then decays once the constraint holds,
arriving at exactly $0$ in four seeds of six, with a final constraint value of $0$ in all six.
In those four seeds training ends on the clean reward once avoidance has been learned, a
transition that a fixed $p$ cannot provide. The other two seeds retain a substantial weight
($alpha = 6.1$, and $16.5$ after a peak of $20.1$ at 127.5k steps), so the penalty is not
guaranteed to switch off within the training budget.
#v(-10pt)
\
However, Lagrangian policies still
show a high per-episode zone visit rate of $83%$, five of six seeds grazing the band edge in
nearly every episode, so what improves is the depth and duration of entries rather than
whether the band is touched at all, and a tighter $epsilon$ would be required to drive the
visit rate towards the $13%$ of $p = 30$.
#v(-10pt)
\
Moreover, $epsilon = 0.01$ was used together with a
dual step size $eta_alpha = 10$ and $alpha_0 = 5$, selected so that $alpha$ can span the
baseline's range of $p$ within the 300 dual updates available; eliminating the sweep over $p$
does not remove the need to scale the dual step appropriately relative to the training budget.
#v(-10pt)
\
== LunarLander
#v(-10pt)
\
This environment changes the reward scale (per-step shaping rewards of order one and
terminal rewards of $plus.minus 100$), the dimensionality of observations and actions, and the
sparse-region geometry: the excluded box $[-0.075, 0.075] times [0.5, 1.0]$ lies above the pad
on the direct descent path. With the Pendulum dual setting unchanged, but with an updated KDE configuration to detect the low desity region right (@sec-protocol), the Lagrangian at $epsilon = 0.01$ reached a clean return of
$258.0 plus.minus 25.7$ compared with $200.1 plus.minus 100.7$ for the pooled sweep
$p in {2, 10, 30}$ (@tab-ll-pooled), strict landing in $96.40 plus.minus 6.4%$ compared with
$71.4 plus.minus 39.3%$ of episodes, and a crash rate of $0.7 plus.minus 1.7%$ compared with
$0.7 plus.minus 1.8%$, at a wall-clock cost of $3719$ s compared with $3728$ s.
The striking point is not a universal return advantage, but the same dual adaptation pattern
seen on the pendulum: the multiplier rises when the constraint is violated and then attenuates
once the policy learns to stay within the covered state space, indicating that the mechanism is
portable across tasks even though the exact optimum trade-off remains task-dependent.
#v(-10pt)
\
The individual baseline settings do reproduce the Pendulum trade-off between avoidance and
task success. Depth-weighted occupancy is way lower for lagr ($0.44%$) than the $p=2$ baseline ($1.054%$). $p=10$ ($0.05%$) and $p=30$ ($0.01%$) beating lagr, while task performance degrades at higher $p$.
#v(-10pt)
\
The multiplier trajectory in @fig-ll-alpha-trajectory follows the Pendulum pattern: $alpha$
rises from $5$ to peaks between $10$ and $40$ while the early policy violates the
constraint, then decays, ending at around $2$ averaged over all six
seeds. The point at which $alpha$ first reaches a low value varies between $300$k and $375$k of the
$500$k training steps, so the penalty is withdrawn per seed once avoidance holds rather than
applied for the full budget.

== Limitations and outlook.
#v(-10pt)
\
The evidence is limited to six seeds per setting in both environments, a single tolerance
$epsilon = 0.01$, and KDE over position as the single type of cost signal. Training in the
true simulator excludes model exploitation by construction, leaving the original motivation
for density penalties untested. The adaptive method also introduces dual hyperparameters
and a transient phase in which $alpha$ can overshoot before settling.

Defining the constraint per training step leaves the per-episode visit rate unbounded, though safety
requirements are frequently phrased in those terms; evaluating such requirements directly would require a
per-episode constraint, or $epsilon$ combined with hard termination on zone entry. The
wider band E3 did not permit a zone-free swing-up, so the auto-tuning claim under
shifted zone geometry remains untested on the pendulum.

Directions for future work are to quantify sensitivity to $epsilon$ and to the number of seeds,
repeating the comparison for several tolerances and more seeds and comparing
against the best single fixed weight rather than the pooled sweep. This would separate
method sensitivity from random variation and test whether the same tolerance can transfer
robustly across runs. A second direction is to improve the density estimator itself by
comparing KDE with alternative cost signals such as k-nearest-neighbor distance estimates,
normalizing-flow density models, or other local density ratios, since the current KDE may smooth
across the excluded zone and can blur the effective boundary with a non-elaborated configuration. A third direction is to impose a
stricter constraint formulation, for example a per-episode constraint, a CVaR-style risk
constraint penalizing with a mean cost over the worst episodes, or $epsilon$ combined with hard termination on zone entry, so that the safety
objective is aligned with the practical requirement rather than with the average per-step cost.
Finally, the learned transition model could be restored for the offline setting, and standard
dual ascent could be replaced by a damped or otherwise stabilized variant @stooke2020 to
shorten the transient and reduce overshoot before the multiplier settles.

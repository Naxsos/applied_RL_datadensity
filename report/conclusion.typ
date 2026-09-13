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
The multiplier trajectory in @fig-alpha-trajectory shows the mechanism that $alpha$ climbs from $5$ to roughly $12.5$ for as
long as the early policy violates the constraint, then decays once the constraint holds,
arriving at exactly $0$ in four seeds of six, with a final constraint value of $0$ in all six.
Training therefore ends on the clean reward once avoidance has been learned, a transition
that a fixed $p$ cannot provide.
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
sparse-region geometry: the excluded box $[-0.2, 0.2] times [0.6, 1.0]$ lies on the direct
descent path to the pad and can be avoided only by a lateral detour. With the Pendulum dual
setting unchanged, the Lagrangian at $epsilon = 0.01$ reached a clean return of
$243.6 plus.minus 63.8$ compared with $233.1 plus.minus 52.4$ for the pooled sweep
$p in {2, 10, 30}$ (@tab-ll-pooled), strict landing in $84.4 plus.minus 33.1%$ compared with
$81.5 plus.minus 25.3%$ of episodes, and a crash rate of $0.08 plus.minus 0.19%$ compared with
$2.4 plus.minus 3.4%$, at a wall-clock cost of $2565$ s compared with $2481$ s.
#v(-10pt)
\
The individual baseline settings do not reproduce the Pendulum trade-off between avoidance and
task success. Depth-weighted occupancy is similar for all three weights ($1.05%$, $1.04%$ and
$1.25%$ for $p = 2$, $10$ and $30$), while task performance degrades at both ends of the sweep:
at $p = 2$ strict landing rates range from $51%$ to $99.5%$ across seeds, and at $p = 30$ one
seed lands in $3%$ of episodes and times out in $97%$. The most consistent fixed setting is
$p = 10$, with strict landing in $93.7 plus.minus 4.6%$ of episodes and a return of
$260.0 plus.minus 17.0$.
#v(-10pt)
\
The multiplier trajectory in @fig-ll-alpha-trajectory follows the Pendulum pattern: $alpha$
rises from $5$ to peaks between $8.7$ and $23.2$ while the early policy violates the
constraint, then decays, ending at exactly $0$ with a final constraint value of $0$ in all six
seeds. The point at which $alpha$ first reaches $0$ varies between $68.5$k and $281$k of the
$300$k training steps, so the penalty is withdrawn per seed once avoidance holds rather than
applied for the full budget.
#v(-10pt)
\
Neither method avoids the box itself. Because it lies on the descent path, Lagrangian and
fixed-weight policies enter it in $25.3%$ and $27.3%$ of evaluation episodes, with
depth-weighted occupancy of $0.95 plus.minus 0.19%$ and $1.11 plus.minus 0.49%$. The Lagrangian
improves the consistency of crash rate and occupancy across seeds rather than the degree of
avoidance, and its mean advantage in return and landing is small relative to the variation
between seeds, as discussed below.

== Limitations and outlook.
#v(-10pt)
\
The evidence is limited to six seeds per setting in both environments, a single tolerance
$epsilon = 0.01$, and KDE over position as the single type of cost signal. Training in the
true simulator excludes model exploitation by construction, leaving the original motivation
for density penalties untested. The adaptive method also introduces dual hyperparameters
and a transient phase in which $alpha$ can overshoot before settling.

On LunarLander, the mean differences between the Lagrangian and the pooled fixed weight in
return ($243.6 plus.minus 63.8$ compared with $233.1 plus.minus 52.4$), strict landing
($84.4%$ compared with $81.5%$), timeout rate and depth-weighted occupancy ($0.95%$ compared
with $1.11%$) lie within one standard deviation across seeds, and the best single weight,
$p = 10$, has the higher mean return ($260.0 plus.minus 17.0$) and strict landing rate
($93.7%$). The Lagrangian means are, however, dominated by one seed that times out in $89%$ of
episodes. This failure is not attributable to the constraint: its $alpha$ first reaches $0$
after $68.5$k steps and averages $0.15$ over the remaining training, so the policy was trained
on an almost unpenalized reward, and timeout rates above $30%$ also occur in 5 of 18
fixed-weight runs. The other five Lagrangian seeds land strictly in $97.5%$ to $100%$ of episodes, and the
median seed exceeds $p = 10$ in return ($269.7$ compared with $264.0$) and strict landing
($99.2%$ compared with $95.0%$). In addition, $p = 10$ is identified as best only after
training the full sweep of 18 runs, whereas the Lagrangian reuses the Pendulum setting
($epsilon = 0.01$, $eta_alpha = 10$, $alpha_0 = 5$) without adjustment, and it is the only
method whose crash rate stays at or below $0.5%$ in every seed (compared with up to $11%$ in
8 of 18 fixed-weight runs). With six seeds, the lander results establish that a single
untuned tolerance is competitive with the best fixed weight of a sweep, at $3%$ higher
wall-clock cost, but not that it outperforms it.

Defining the constraint per training step leaves the per-episode visit rate unbounded, though safety
requirements are frequently phrased in those terms; evaluating such requirements directly would require a
per-episode constraint, or $epsilon$ combined with hard termination on zone entry. In
LunarLander, the box lies on the direct descent path, and both methods enter it in about
a quarter of evaluation episodes ($25%$ compared with $27%$). The
wider band E3 did not permit a zone-free swing-up, so the auto-tuning claim under
shifted zone geometry remains untested on the pendulum.

Directions for future work are to evaluate several tolerances and more seeds on LunarLander
and compare them against the best single fixed weight rather than the pooled sweep, to
restore the learned transition model, and to replace standard dual ascent with a damped
variant @stooke2020 in order to shorten the transient.

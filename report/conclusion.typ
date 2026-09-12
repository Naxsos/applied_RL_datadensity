= Conclusion <sec-conclusion>
#v(-10pt)
\
This report compared a fixed density-penalty coefficient $p$ @lantz2025 with a Lagrange
multiplier learned online from a user-specified tolerance $epsilon$. Under the shared
experimental design, adaptive weighting improved the safety-performance trade-off on
Pendulum and reduced over-penalization on LunarLander. It replaces the fixed-weight sweep
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
This environment changes the reward scale (per-step shaping rewards of order
one and terminal rewards of $plus.minus 100$) and the sparse-region geometry (low altitudes
rather than an excluded band). Its constraint concerns unstable near-ground states that
landing trajectories must approach. At $p = 10$ or $p = 30$ the lander does not land, with $0%$ strict landings,
timeout rates of $87%$ and $92%$, and clean returns of $20 plus.minus 11$ and
$-6 plus.minus 14$, because the density penalty on low altitude outweighs the landing reward.
At $p = 2$ the outcome splits across seeds, with strict landing rates of $0%$, $57%$ and
$100%$. The unpenalized reference $p = 0$ lands in $67 plus.minus 40%$ of episodes for a return
of $203 plus.minus 91$.
#v(-10pt)
\
The Lagrangian at $epsilon = 0.05$ lands in $98.3 plus.minus 2.4%$ of
episodes with no crashes and a return of $267 plus.minus 4$ over three seeds, and
$epsilon = 0.1$ performs similarly at $96.7 plus.minus 2.1%$ and $265 plus.minus 10$; pooling
$epsilon in {0.02, 0.05, 0.1}$ compared with $p in {2, 10, 30}$ gives strict landing rates of $82%$
compared with $17%$ and returns of $227 plus.minus 76$ compared with $56 plus.minus 91$.
#v(-10pt)
\
Policies that land occupy the corridor for only a few percent of their training steps, between $1.4%$ and $6.6%$, so a tolerance of $0.05$ or $0.1$ is met early,
$alpha$ decays to $0$ in five seeds of six, and the agent resumes optimization of the clean landing objective.
A tolerance of $0.02$, by contrast, is close to the minimum occupancy required for landing. This means, $alpha$
stabilizes near its initial value and the outcome splits across seeds much as $p = 2$ does,
at $0%$, $57%$ and $95%$.
#v(-10pt)
\
These results demonstrate reduced over-penalization, but do not establish improved safety. Landing
policies register per-episode corridor visit rates averaging $85%$ to $93%$, since a brief
touchdown already satisfies the corridor's speed or tilt thresholds, while the low visit rates
of $p = 10$ and $p = 30$ belong to policies that hover and never land. Fixed weight and
Lagrangian are separated in this environment by the over-penalization failure mode rather than
by avoidance, and the relevant safety assessment is therefore per step: policies
trained at $epsilon = 0.05$ finish at corridor step rates of $1.4%$ to $2.6%$, within the
specified tolerance.

== Limitations and outlook.
#v(-10pt)
\
The evidence is limited to six seeds per setting for Pendulum,
three for LunarLander, with KDE over position as the single type of cost signal. Training in the
true simulator excludes model exploitation by construction, leaving the original motivation
for density penalties untested. The adaptive method also introduces dual hyperparameters
and a transient phase in which $alpha$ can overshoot before settling.

Defining the constraint per training step leaves the per-episode visit rate unbounded, though safety
requirements are frequently phrased in those terms; evaluating such requirements directly would require a
per-episode constraint, or $epsilon$ combined with hard termination on zone entry. The
wider band E3 did not permit a zone-free swing-up, so the auto-tuning claim under
shifted zone geometry remains untested on the pendulum, and in LunarLander the excluded region
was not excluded from the data, so the density signal and the constraint quantify
different properties.

Directions for future work are to repeat the lander experiment with a region
that is both excluded from the data and avoidable, to restore the learned transition model,
and to replace standard dual ascent with a damped variant @stooke2020 in order to shorten the
transient.

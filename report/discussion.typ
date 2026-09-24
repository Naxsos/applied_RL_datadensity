= Discussion <sec-discussion>

== How a learned multiplier compares to a fixed one
#v(-10pt)
\
The pendulum comparison in @tab-pendulum-pooled and @fig-alpha-trajectory answers the first research question of @sec-background affirmatively for the range tested: a single Lagrangian setting matches or beats the entire fixed-weight sweep without running the sweep, through the mechanism the constrained formulation predicts.
Early in training, while the still-mostly-random policy violates $C <= epsilon$ by an order of magnitude, dual ascent @eq:dual-ascent raises $alpha$ from $alpha_0 = 5$ to a mean peak of $12.5$, between the baseline's $p=10$ and $p=30$, supplying a penalty magnitude the fixed sweep had to guess at rather than the value a practitioner happens to try first.
Once the policy learns to route around the band and $C$ falls back under $epsilon$, $alpha$ decays, reaching $0$ in four of six seeds, so the back half of training optimizes the clean reward instead of paying a penalty the constraint no longer requires.
Because the point at which a given seed's policy actually learns to avoid the band varies, so does this decay: $alpha$ keeps rising for as long as that run still violates the constraint and only falls once avoidance is learned in it, which is why four of six seeds reach $0$ while the remaining two, whose policies keep drifting back toward the band late in training, retain a substantial weight: seed 1 ends at $alpha = 6.1$, and seed 4 peaks at $20.1$ after 127.5k steps and still ends at $16.5$, above $p=10$.
The adaptation is therefore per run, but not guaranteed to switch the penalty off within the training budget.
A fixed $p$ has no such per-run flexibility: whichever value is chosen is paid for all 150k steps regardless of how quickly that seed's policy learns to avoid the band, which is why $p=30$, tuned tightly enough to nearly eliminate zone entries, is also the only setting that fails the task outright in some cases.
#v(-10pt)
\
That comparison, however, pools all three baseline settings against the single Lagrangian setting, so part of the reported margin reflects that two of the three pooled values are simply not the ones a sweep should end up selecting.
@tab-pendulum-by-p in the appendix breaks the pool back out into $p=2$, $p=10$, and $p=30$, and against these individual settings lagr does not win a single metric outright: $p=30$ reaches a lower zone depth-weighted step fraction ($0.0005%$ against lagr's $0.075%$), and $p=10$ edges out a marginally better return and a marginally faster swing-up.
What lagr wins is the combination: it matches $p=2$ and $p=10$'s perfect upright success rate, which $p=30$ fails to reach, while still cutting zone depth to a fraction of $p=2$'s and $p=10$'s ($10.3 times$ and $2.5 times$ lower respectively).
The honest reading, then, is not that lagr dominates the sweep on any one axis, but that it is the only setting simultaneously close to the best at consistently finishing the task and close to the best at avoiding the zone, where every individual $p$ trades one of those against the other.
What it buys is a favorable trade-off reached without knowing in advance which $p$ produces it, rather than a categorically safer or better-performing policy than the best possible fixed weight on any single axis.
That return gap is itself smaller than either group's standard deviation ($-331.2 plus.minus 9.5$ for lagr against $-337.1 plus.minus 7.1$ pooled), so "marginally better" return is compatible with no real difference at all: what the comparison actually establishes is the safety metrics and their consistency across seeds, not a return advantage.


== A shared limit: none of the methods fully avoid the zone
#v(-10pt)
\
Across every method tested, including $p=30$ and lagr, the per-episode zone visit rate never reaches zero: even @fig-vis-trajectories's shallowest trajectories still cross the red boundary at the edge of the band before turning back, rather than staying clear of it entirely.
A plausible explanation lies in the density estimate rather than in either method's policy.
Our collection protocol drops a transition only when its successor state lies inside the band and lets the episode continue (@sec-protocol), unlike the terminating protocol of @lantz2025. States directly outside both band edges are therefore densely covered, and a KDE of bandwidth $0.1$ smooths this density into the band.
The resulting penalized region is not the band itself, and it is not symmetric around it: under the fitted KDE, $hat(rho) < 0.025$ holds for $theta$ from about $66 degree$ to $101 degree$, so the cost is $1$ on roughly $6 degree$ outside the lower edge and $0$ on the last $7 degree$ of the band next to $108 degree$, the edge facing the start state $theta = pi$.
A policy swinging up from rest reaches exactly this unpenalized sliver first, and is not penalized for occupying it even though those states are technically inside the band. This matches @fig-vis-trajectories, where lagr's zone entries cluster at exactly that edge.
The zone depth-weighted step fraction reported in @tab-pendulum-pooled and @tab-pendulum-by-p is built to discount exactly this: weighting entries toward zero near the edge and toward one at the band's centre (@sec-metrics), it scores a shallow graze far below a genuine crossing, which is why $p=30$ and lagr both still register some zone contact yet post depth-weighted values close to zero.
That weighting compensates for the smoothing at the level of the reported metric, but it does not remove the underlying limitation: a KDE-derived cost signal has a soft boundary by construction, so no penalty scheme built on top of it, fixed or learned, should be expected to drive raw zone contact to exactly zero.

== Generalizing to LunarLander
#v(-10pt)
\
LunarLander changes several things at once relative to the pendulum: a landing task rather than a swing-up, an 8-dimensional observation against the pendulum's 3, and a two-dimensional continuous thrust action against a single continuous torque, so whatever transfers here is not simply a restatement of the pendulum result on a relabeled environment.
#v(-10pt)
\
Pooled against the entire baseline sweep, @tab-ll-pooled shows lagr with the higher true return, a higher strict landing rate, a lower timeout rate, and a comparable crash rate and depth-weighted zone fraction. That pooled comparison is somewhat flattering, though, because it mixes in the weak $p=30$ setting and therefore pulls the baseline average down.
The more relevant comparison is against the individual fixed settings. In @tab-ll-by-p, the fixed $p=10$ and $p=30$ runs do not consistently solve the landing task: both settings show worse landing-rate and timeout outcomes than the best-performing runs, and their variance is substantially larger than lagr's on the task metrics. This is the key point from the metrics: the more conservative fixed weights improve the zone-depth behavior, but they do so by sacrificing task consistency rather than by reliably solving the landing problem.
#v(-10pt)
\
The main advantage of lagr is therefore not a universal return win, but its better compromise between task reliability and zone avoidance. Compared with $p=2$, lagr keeps a much lower depth-weighted zone fraction while remaining competitive on landing outcomes. The problem is not that all fixed weights fail to avoid the zone; it is that $p=2$ is too weak to learn the avoidance behavior reliably, whereas the more conservative settings do avoid it better but at the cost of landing consistency and task completion. This is consistent with @fig-ll-combined-traj, where lagr keeps trajectory density concentrated near the landing pad while still staying farther from the excluded box than $p=2$, and where $p=10$ and $p=30$ become more conservative but less reliable on the landing objective.
#v(-10pt)
\
The tuning mechanism itself behaves as on the pendulum: @fig-ll-alpha-trajectory shows $alpha$ rising while $C > epsilon$ early in training and decaying once $C$ stays below $epsilon$, with the mean value settling back toward about $2$ by the end of training. The same adaptation pattern recurs on a different task, even though the resulting trade-off is not the same as on the pendulum.
That is the more portable result. The auto-tuned multiplier generalizes across environments, while the exact task-specific balance between safety and performance remains environment-dependent.
#v(-10pt)
\

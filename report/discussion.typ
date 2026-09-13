= Discussion <sec-discussion>

== How a learned multiplier compares to a fixed one
#v(-10pt)
\
The pendulum comparison in @tab-pendulum-pooled and @fig-alpha-trajectory answers the first research question of @sec-background affirmatively for the range tested: a single Lagrangian setting matches or beats the entire fixed-weight sweep without running the sweep, through the mechanism the constrained formulation predicts.
Early in training, while the still-mostly-random policy violates $C <= epsilon$ by an order of magnitude, dual ascent @eq:dual-ascent raises $alpha$ past the baseline's largest tested value ($12.5$ vs. $p=10$), supplying exactly the penalty magnitude the fixed sweep had to guess at rather than the value a practitioner happens to try first.
Once the policy learns to route around the band and $C$ falls back under $epsilon$, $alpha$ decays, reaching $0$ in four of six seeds, so the back half of training optimizes the clean reward instead of paying a penalty the constraint no longer requires.
Because the point at which a given seed's policy actually learns to avoid the band varies, so does this decay: $alpha$ keeps rising for as long as that run still violates the constraint and only falls once avoidance is learned in it, which is exactly why four of six seeds reach $0$ while the remaining two, whose policies keep drifting back toward the band late in training, settle on a small residual weight instead.
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
The pendulum dataset only drops transitions once the angle actually enters the band, and each collection episode terminates exactly at that crossing (@sec-background), so transitions naturally accumulate as trajectories approach the boundary from outside, which is exactly where a KDE of bandwidth $0.1$ has the most density to smooth across.
The estimated density $hat(rho)$ does not fall to zero right at the boundary; it decays gradually, so a thin margin just inside the true excluded band can still sit above the $0.025$ threshold, and a policy is not penalized for occupying a state the cost signal has not yet flagged, even though that state is technically inside the band.
The zone depth-weighted step fraction reported in @tab-pendulum-pooled and @tab-pendulum-by-p is built to discount exactly this: weighting entries toward zero near the edge and toward one at the band's centre (@sec-metrics), it scores a shallow graze far below a genuine crossing, which is why $p=30$ and lagr both still register some zone contact yet post depth-weighted values close to zero.
That weighting compensates for the smoothing at the level of the reported metric, but it does not remove the underlying limitation: a KDE-derived cost signal has a soft boundary by construction, so no penalty scheme built on top of it, fixed or learned, should be expected to drive raw zone contact to exactly zero.

== Generalizing to LunarLander
#v(-10pt)
\
LunarLander changes several things at once relative to the pendulum: a landing task rather than a swing-up, an 8-dimensional observation against the pendulum's 3, and four discrete thrusters against a single continuous torque, so whatever transfers here is not simply a restatement of the pendulum result on a relabeled environment.
#v(-10pt)
\
Pooled against the entire baseline sweep, @tab-ll-pooled shows lagr ahead on six of seven metrics, but that comparison flatters it: $p=30$, the weakest setting in the pool (lowest return, highest zone depth-weighted fraction of all four groups), drags the pooled baseline average down and inflates lagr's apparent margin over it.
@tab-ll-by-p, which breaks the pool back into $p=2$, $p=10$, and $p=30$, tells a less favorable story: $p=10$ alone is the strongest setting on every task-performance metric measured, reaching the highest return ($260.0 plus.minus 17.0$ against lagr's $243.6 plus.minus 63.8$), landing success ($99.5 plus.minus 0.9%$ against $97.9 plus.minus 4.7%$), strict landing rate ($93.7 plus.minus 4.6%$ against $84.4 plus.minus 33.1%$), and lowest timeout rate ($4.2 plus.minus 2.6%$ against $15.5 plus.minus 32.8%$), with markedly less run-to-run variance than lagr, whose own standard deviation on strict landing and timeout rate exceeds its mean.
Against $p=10$ specifically, lagr does not win a single one of these metrics.
#v(-10pt)
\
The one result that survives this stricter, per-$p$ comparison is the crash rate: lagr's $0.1 plus.minus 0.2%$ sits an order of magnitude below every individual baseline setting ($2.6%$, $2.2%$, and $2.3%$ for $p=2$, $p=10$, and $p=30$), a gap larger than any of the groups' own spread.
@fig-ll-combined-traj is consistent with this: the baseline panels, most visibly $p=2$ and $p=10$, show long looping excursions to the side of the pad before landing, while lagr's trajectories stay more tightly clustered along a direct descent, a plausible behavioral source of its lower crash rate even though, as the next paragraph shows, it does not visibly avoid the excluded box any more than the baseline does.
With only six seeds per setting, this single robust effect, rather than a broad performance or safety advantage, is the realistic summary of what generalizes to LunarLander: lagr trades a modest amount of task performance relative to a well-tuned $p=10$ for a substantially lower crash rate, without needing to already know that $p=10$ is the setting worth tuning toward.
#v(-10pt)
\
The tuning mechanism itself does transfer even where its effect on the outcome is mixed: @fig-ll-alpha-trajectory shows $alpha$ rising while the policy still enters the sparse box and decaying once violations subside, the same pattern @fig-alpha-trajectory establishes for the pendulum.
That the online-tuning behavior reproduces cleanly on a task this different, while the resulting safety-performance trade-off does not clearly beat the best fixed $p$, suggests the auto-tuning mechanism is the more portable property of the Lagrangian formulation, not any particular trade-off it happens to buy on a given task.
#v(-10pt)
\
None of the methods visibly avoid the LunarLander zone either.
The pendulum trajectories in @fig-vis-trajectories showed a visible avoidance pattern: $p=10$, $p=30$, and lagr all leave the excluded wedge visibly emptier than $p=2$ does.
@fig-ll-combined-traj shows no equivalent effect for LunarLander: all four panels, $p in {2, 10, 30}$ and lagr alike, show trajectory density passing through and around the excluded box at comparable intensity, with none of the four settings leaving it visibly emptier than the others.
This matches the depth-weighted step fraction in @tab-ll-pooled and @tab-ll-by-p, which sits in a narrow band across all four groups ($0.953%$ to $1.248%$) rather than spreading out the way it does across the pendulum sweep ($0.076%$ to $0.760%$); if the penalty were shaping the route the way it does on the pendulum, a larger $p$ should show at least a directional reduction, whereas $p=30$ instead posts the highest depth-weighted fraction of the four groups.
#v(-10pt)
\
The pendulum's soft-boundary explanation (@sec-metrics's edge decay of the KDE density) does not obviously account for this: it explains why a policy that is otherwise avoiding a zone still grazes its edge, not why every penalty weight tested, including the largest, fails to produce any visible avoidance at all.
A more likely explanation is geometric: the box sits inside the funnel every landing trajectory must pass through en route to the pad (@fig-ll-offline-zone), and that same corridor is also the most direct and stable approach to the platform, so steering around the box trades stability for avoidance rather than getting both for free the way the pendulum's alternate swing-up side does; no penalty magnitude tested changes this qualitative trade.On the evidence collected, the honest conclusion is that additional tuning may be needed before the LunarLander zone can be treated as a fair test of avoidance at all, rather than that density-based avoidance transfers to this environment as-is.

#import "@preview/dashy-todo:0.1.3": todo

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

#todo("Add lunar lander part of discussion")

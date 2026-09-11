= Introduction

Model-based reinforcement learning agents that are trained or evaluated offline rely on a
learned transition model to stand in for the real environment. That model is only accurate
where it has seen enough data: outside those regions its predictions are unreliable, and a
policy optimized against it can learn to exploit exactly the states where the model is wrong,
producing behavior that looks good under the learned dynamics but fails, or is unsafe, under
the true dynamics. This is especially problematic when parts of state space are inherently
under-sampled in the offline dataset -- not due to a data collection oversight, but because
those regions are rare, costly, or dangerous to visit in the first place.

A standard mitigation is to penalize the reward with a data-density cost signal: states or
state-action pairs that are far from the training distribution incur an additional cost,
discouraging the policy from visiting them. #emph[(TODO: cite the reference paper, ES2025-194,
once entered in refs.bib)] scales this penalty by a fixed coefficient $p$, chosen by hand and
held constant for the duration of training. This works, but the coefficient carries all of the
burden: too small, and the policy still routes through the excluded, low-data region; too
large, and the penalty dominates the task objective and cripples performance elsewhere. Because
the "right" value of $p$ depends on the environment, the task reward's scale, and the shape of
the excluded region, it must be re-tuned by hand whenever any of those change, and it cannot
adapt over the course of training as the policy's visitation distribution shifts.

We instead treat staying out of the low-data region as a constraint rather than as an
unconstrained penalty term, and cast the problem as a constrained Markov decision process. The
penalty weight becomes a Lagrange multiplier, updated online via dual ascent so that it grows
when the policy violates the constraint more often than allowed and shrinks otherwise. Under
this formulation, the practitioner no longer chooses a penalty magnitude directly; instead they
specify an *allowed violation rate* $epsilon$ -- a quantity with a direct operational
meaning -- and the weight is learned to enforce it. We compare this Lagrangian weighting scheme
against the fixed-weight baseline under otherwise identical conditions: the same data-density
cost signal, the same agent, and the same evaluation protocol, so that any difference in
outcome is attributable to the weighting scheme alone.

== Contributions Gerrit

- A shared, pluggable reward-wrapper implementation in which the baseline and Lagrangian
  conditions differ *only* in the weight component -- same cost signal, same agent, same
  evaluation protocol -- so the comparison isolates a single design choice.
- An empirical comparison of fixed-weight and Lagrangian-weight penalization on a pendulum
  swing-up task with an excluded, low-data zone, including how each method's performance and
  safety trade-off responds to its own hyperparameter sweep.
- A generalization test on LunarLander, checking that any advantage observed on the pendulum
  task is not an artifact of its particular geometry or excluded-zone construction.
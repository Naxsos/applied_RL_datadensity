= Methodology

== Environments Jonas: Lunar Lander; Maram: Pendulum
- *Primary: Pendulum swing-up with excluded zone.* One-sided low-data band the agent must
  learn to avoid while still reaching upright (env E1); shifted/relabeled variant (env E3) to
  test whether Lagrangian auto-tuning survives a change in the "correct" $p$.
- *Generalization: LunarLander.* #emph[TODO: define the excluded-zone / low-data-region
  analogue for this environment, and why it is expected to stress-test the same claim.]
- Shared components held fixed across both methods per environment: transition model /
  simulator, SAC hyperparameters, training steps, seeds.

== Methods compared Maram 
- `baseline`: KDE density cost + fixed weight, swept over $p in {dots}$.
- `lagr`: KDE density cost + Lagrangian weight, swept over $epsilon in {dots}$.
- Ensemble- and BNN-based uncertainty cost signals are out of scope for this paper.

== Training and evaluation protocol Maram 
- Offline dataset generation per environment.
- SAC training, $N$ steps, $>=5$ seeds per cell, mean $plus.minus$ std reported.
- Evaluation always on the clean, unpenalized reward, so methods with different training
  objectives remain comparable.
- Fixed start state, fixed episode length, greedy (deterministic) action evaluation.

== Metrics Theresa 
- *Safety (lower is better):* zone visit rate (episode-level), zone step fraction, left/right
  path split (route-around behavior).
- *Task performance (higher is better):* true return mean/std, upright success rate,
  time-to-upright.
- *Constraint behavior (Lagrangian only):* final $alpha$, $alpha$ trajectory over training,
  constraint value $C$ vs. $epsilon$, constraint-satisfied flag.
- *Robustness / tunability:* spread of (return, zone rate) across each method's own knob sweep;
  a flat spread means the method is easy to tune -- the Lagrangian's expected advantage.
- *Compute:* wall-clock training time, where relevant to a claim.
- For LunarLander: same categories, redefined for that environment's task-success and
  "excluded zone" analogue.
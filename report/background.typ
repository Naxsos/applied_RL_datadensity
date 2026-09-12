= Background and Problem Formulation <sec-background>

== Learning a policy from a fixed dataset

No interaction with the real system is permitted while the policy is being learned. Instead, the available information consists of a dataset $cal(D) = {(s_i, a_i, s'_i)}_(i=1)^N$ of transitions recorded beforehand, in industrial applications typically as ordinary process data. It is used to fit a state-transition model $hat(f)(s, a) approx s'$ that replaces the real system inside a simulated environment, and to train a policy $pi$ against that simulation @levine2020offline. The reliability of this approach depends on the coverage of the dataset. In regions where $cal(D)$ is sparse, $hat(f)$ extrapolates, and policy optimization may exploit these extrapolations by identifying state-action sequences with high predicted returns that are not reproducible in the real system.

The standard countermeasure in offline model-based RL is to subtract an estimate of the model's own uncertainty from the reward
@yu2020mopo @kidambi2020morel. Lantz et al. @lantz2025 replace this estimate with a computationally less expensive proxy. They quantify the data
support for a state by fitting a _density_ model to the offline dataset and penalizing the policy for entering sparsely covered regions. Two arguments support the substitution. Data coverage supports model accuracy, and safety-critical deployments motivate avoiding unobserved configurations even when their predicted returns are high.

== The reference approach: a fixed-weight density penalty

The experiments in @lantz2025 use the Gymnasium `Pendulum-v1` swing-up task @towers2024gymnasium, into which a sparsely covered region is introduced artificially. Every state whose angle satisfies $theta in [2 pi slash 5, 3 pi slash 5]$ is excluded from the dataset. 2000 episodes of at most 200 steps are rolled out with uniformly random torques,each episode terminates upon entry into that band, and the remaining 155,000 transitions constitute the offline dataset. An LSTM with 50 units and a dense output layer, fed by a 4-step sliding window, serves as the transition model. A kernel density estimate @chen2017kde with bandwidth 0.1 is fitted to the positional encoding $(cos theta, sin theta)$ of the visited states. The density model enters training as a constant penalty $p$ applied whenever the estimated density $hat(rho)$ drops below a threshold of 0.025, $ r_p (s, u, s') = underbrace(-(theta^2 + 0.1 dot(theta)^2 + 0.001 u^2), "original reward") - p dot bb(1)[hat(rho)(s') < 0.025], $ <eq:paper-reward>

and a Soft Actor-Critic agent @haarnoja2018 is trained on it for 150,000 steps starting from the resting state $theta = pi$, $dot(theta) = 0$. Since the excluded band blocks one side of the swing only, the pendulum can still be raised by swinging up the other way, and the share of evaluation episodes that swing up through the penalized side is the study's primary safety metric. For over 2000 episodes it falls from 41% at $p = 2$ to 36% at $p = 10$ and 15% at $p = 30$.

Two findings motivate the present study are firstly the fact that avoidance was not reached within the training budget even at the largest penalty and furthermore, as the authors note, the weight of the penalty term must be selected according to the environment's reward function. Hence, the original per-step reward in @eq:paper-reward lies in $[-16.27, 0]$, which makes $p = 2$ a small penalty and $p = 30$ nearly twice the worst per-step reward obtainable, so these three values would yield entirely different trade-offs in an environment whose rewards are scaled differently.

== Problem formulation

Consider a Markov decision process $(cal(S), cal(A), P, r, gamma)$ over continuous states and actions, accompanied by a fixed offline dataset $cal(D)$ from which a _cost signal_ $c : cal(S) -> [0, 1]$ is derived once and thereafter held constant. In this report $c$ is always a function of the KDE density of the offline data: it vanishes on well-covered states and increases towards 1 as coverage decreases, with the precise mapping for each environment stated in @sec-protocol. Both methods studied here optimize the same penalized reward, 

$ r_w (s_t, a_t, s_(t+1)) = r(s_t, a_t) - w_t dot c(s_(t+1)), $ <eq:penalized-reward>

in which $r$ denotes the clean task reward, the cost is evaluated at the successor state, and $w_t >= 0$ is the penalty weight. The two methods differ only in the choice of $w_t$. Writing

$ J(pi) = EE_pi [sum_t gamma^t r(s_t, a_t)],
  quad quad
  C(pi) = EE_pi [1/T sum_(t=1)^T c(s_t)] $

for the task return and the expected per-step cost over an episode of length $T$, the quantity $C(pi) in [0, 1]$ measures the share of time the policy spends in penalized states, weighted by the severity of the penalty there.

*Fixed weight (baseline).* The reference method fixes $w_t equiv p$ at a manually selected constant, leaving the agent to maximize $EE_pi [sum_t gamma^t (r_t - p dot c_(t+1))]$. This is one particular scalarization of return against cost, and the cost level $C(pi_p^*)$ that
results is determined by that scalarization. The mapping $p |-> C(pi_p^*)$ is monotone but not known in advance, so achieving a target cost level requires a sweep over $p$, which must be repeated whenever the reward scale, the shape of the sparse region or the episode horizon changes.

Furthermore, the weight acts open-loop for the entire run, with no reference to whether the requirement is currently being met. A $p$ that is too small allows the policy to traverse the sparse region because the reward lost by avoiding it exceeds the incurred penalty; a $p$ that is too large causes the penalty to dominate the
reward, and the agent abandons the task rather than risk a violation. Our experiments produce both regimes (@sec-results).

*Constrained formulation (this work).* 
We formulate the requirement as a constraint within a constrained MDP @altman1999, $ max_pi J(pi) quad "subject to" quad C(pi) <= epsilon, $ <eq:cmdp> in which $epsilon in [0, 1]$ is the largest per-step cost the practitioner is willing to accept. Unlike $p$, the tolerance $epsilon$ specifies an acceptable constraint level independently of the task reward scale. For a binary cost, it represents the maximum permitted fraction of training steps spent in penalized states.

Relaxing @eq:cmdp in the Lagrangian sense,
$ min_(alpha >= 0) max_pi L(pi, alpha), quad quad
  L(pi, alpha) = J(pi) - alpha (C(pi) - epsilon), $ <eq:lagrangian>
converts the constraint back into a penalty, but one whose weight $alpha$ is now subject to optimization itself. Holding $alpha$ fixed reduces the inner problem to ordinary RL on the reward $r - alpha c$, which is @eq:penalized-reward with $w_t = alpha$; the outer problem is handled by projected gradient descent on $alpha$, whose gradient $partial L slash partial alpha = -(C(pi) - epsilon)$ yields $ alpha <- max(0, alpha + eta_alpha (hat(C) - epsilon)), $ <eq:dual-ascent> with $hat(C)$ an empirical estimate of $C(pi)$ taken over the most recent training steps and $eta_alpha$ the dual step size.

Interpreted as a controller, the update increases the penalty weight when the policy violates the constraint ($hat(C) > epsilon$), increasing the incentive to avoid the penalized region, and lowers it again once the constraint holds. Complementary slackness at a saddle point of @eq:lagrangian requires $alpha^* (C(pi^*) - epsilon) = 0$: where the constraint is inactive at the optimum, the multiplier is zero and training proceeds on the clean reward, so the penalty deactivates itself when it is not needed. Combining an off-the-shelf actor-critic for the policy step with a slower dual step is a conventional Lagrangian approach in safe RL @tessler2019 @ray2019; @stooke2020 study its transient behaviour and propose PID-controlled variants. We retain the standard form of @eq:dual-ascent, since the study focuses on the effect of learning the penalty weight rather than on tuning the dual controller.

The constrained formulation introduces $eta_alpha$, an initial value $alpha_0$ and an update interval of its own as hyperparameters, and the way $hat(C)$ is measured fixes what $epsilon$ actually denotes, since bounding the average of a continuous cost signal and bounding the share of steps inside a designated region are distinct constraints defined on different scales. We implement both variants and record in @sec-protocol which one each experiment uses.

== Research question and experimental design

We investigate whether replacing the constant $p$ with a learned multiplier $alpha$:
- achieves safety comparable to the best manually tuned $p$ without a sweep over $p$;
- avoids over-penalization that prevents task completion; and
- maintains these properties in a second environment with a different reward scale and
  sparse-region geometry, where the original $p$ values are not expected to transfer.

To isolate the weighting scheme, both methods share the offline dataset, KDE cost signal, simulator, SAC agent and hyperparameters, training budget, seeds, and clean-reward evaluation. Unlike Lantz et al., who train within a learned LSTM transition model, we train in the true simulator and derive only the penalty from offline data. This removes model error as a confounder but prevents conclusions about either method's exploitation of a learned model.

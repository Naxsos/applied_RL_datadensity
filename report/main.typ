#import "@preview/ieee-monolith:0.1.0": ieee

#show: ieee.with(
  title: [*Lagrangian Weighting for Data-Density-Aware Offline Reinforcement Learning*],
  abstract: [
    Model-based offline reinforcement learning policies can exploit regions of state-action
    space where the learned transition model is unreliable due to sparse training data. A
    common mitigation penalizes the reward by a data-density cost signal, scaled by a fixed
    weight that must be hand-tuned per environment. We instead treat the density constraint as
    a constrained MDP and learn the penalty weight online via Lagrangian dual ascent, so the
    practitioner specifies an allowed constraint-violation rate instead of an arbitrary penalty
    magnitude. We compare this Lagrangian weighting scheme against the fixed-weight baseline on
    a pendulum swing-up task with an excluded low-data zone, and test generalization on
    LunarLander. #emph[TODO: fill in with final headline result once experiments are complete.]
  ],
  authors: (
    (
      name: "Theresa Geber, Maram Hadhri, Jonas Lang, Gerrit Grätz",
      department: [Applied Reinforcement Learning],
      organization: [Ludwig Maximilian University],
      location: [Munich, Germany],
    ),
  ),
  index-terms: ("Reinforcement learning", "Offline RL", "Lagrangian optimization", "Constrained MDP", "Data density"),
  bibliography: bibliography("refs.bib"),

  global-font: ("Charter", "Hiragino Kaku Gothic Interface"),
)


#outline(indent: auto)
#set page(numbering: "1 / 1",)




#include "introduction.typ"

#include "background.typ"

#include "methodology.typ"

#include "results.typ"

#include "discussion.typ"

#include "conclusion.typ"

#include "appendix.typ"












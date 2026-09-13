#import "@preview/charged-ieee:0.1.4": ieee

#set page(
  numbering: "1",
  footer: context {
    if counter(page).get().first() > 0 {
      align(center, counter(page).display())
    }
  }
)

#show: ieee.with(
  title: [Lagrangian Weighting for Data-Density-Aware Offline Reinforcement Learning],
  abstract: [
Model-based offline reinforcement learning policies can exploit regions of state-action
    space where the learned transition model is unreliable due to sparse training data. A
    common mitigation penalizes the reward by a data-density cost signal, scaled by a fixed
    weight that must be hand-tuned per environment. We instead treat the density constraint as
    a constrained MDP and learn the penalty weight online via Lagrangian dual ascent, so the
    practitioner specifies a tolerance on the expected density cost instead of an arbitrary
    penalty magnitude. We compare this Lagrangian weighting scheme against the fixed-weight baseline on
    a pendulum swing-up task with an excluded low-data zone, and test generalization on
    LunarLander. On the pendulum, a single Lagrangian setting matches the best results of the
    entire fixed-weight sweep without needing to search over $p$; on LunarLander, it lowers the
    crash rate but falls short of the best fixed weight's task performance.
  ],
  authors: (
    (
      name: "Theresa Geber",
      organization: [Ludwig-Maximilians-Universität],
      // location: [Munich, Germany],
    ),
    (
      name: "Maram Hadhri",
      organization: [Ludwig-Maximilians-Universität],
      // location: [Munich, Germany],
    ),
    (
      name: "Jonas Lang",
      organization: [Ludwig-Maximilians-Universität],
      // location: [Munich, Germany],
    ),
    (
      name: "Gerrit Grätz",
      organization: [Ludwig-Maximilians-Universität],
      // location: [Munich, Germany],
    ),
    
  ),
  index-terms: ("Offline reinforcement learning", "Safe reinforcement learning", "Constrained MDP", "Lagrangian methods", "Kernel density estimation"),
  bibliography: none,
  figure-supplement: [Fig.],
)



#include "introduction.typ"

#include "background.typ"

#include "methodology.typ"

#include "results.typ"

#include "discussion.typ"

#include "conclusion.typ"

= References
#bibliography("refs.bib")
#counter(heading).update(0)
#include "appendix.typ"

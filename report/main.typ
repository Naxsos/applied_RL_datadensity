#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/dashy-todo:0.1.3": todo

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
    practitioner specifies an allowed constraint-violation rate instead of an arbitrary penalty
    magnitude. We compare this Lagrangian weighting scheme against the fixed-weight baseline on
    a pendulum swing-up task with an excluded low-data zone, and test generalization on
    LunarLander. #todo[TODO: fill in with final headline result once experiments are complete.]
  
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
  index-terms: ("Affective AI", "LLM", "VLM", "Intention Detection"),
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

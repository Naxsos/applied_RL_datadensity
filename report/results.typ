= Results Jonas: Lunar Lander; Gerrit: Pendulum <sec-results>

== Pendulum (E1): baseline vs. Lagrangian

#let rows = csv("group_comparison.csv", row-type: dictionary)

// per-metric display: whether it's a rate (scale x100, "%" suffix) and how
// many decimals to show -- keyed by the exact label written into the CSV by
// scripts/report_group_comparison.py
#let display = (
  "Zone depth-weighted step fraction": (label: [Zone depth-weighted step fraction], percent: true, decimals: 3),
  "True return": (label: [True return], percent: false, decimals: 1),
  "Upright success rate": (label: [Upright success rate], percent: true, decimals: 1),
  "Time to upright (steps)": (label: [Time to upright (steps)], percent: false, decimals: 1),
  "Wall-clock training time (s)": (label: [Wall-clock training time (s)], percent: false, decimals: 1),
)

#let fmt-cell(mean, std, spec) = {
  let scale = if spec.percent { 100 } else { 1 }
  let unit = if spec.percent { "%" } else { "" }
  let m = calc.round(float(mean) * scale, digits: spec.decimals)
  let s = calc.round(float(std) * scale, digits: spec.decimals)
  [#m $plus.minus$ #s#unit]
}

// group ids come straight from the CSV columns (method_knobvalue, e.g.
// "baseline_p2", "lagr_epsilon0.01") so adding another swept value or method
// to scripts/report_group_comparison.py's input just adds a column here --
// no hardcoded method list to keep in sync.
#let group-ids = rows.first().keys().filter(k => k.ends-with("_mean")).map(k => k.slice(0, k.len() - 5))

#let pretty-label(gid) = {
  if gid == "baseline" {
    [Baseline ($p in {2, 10, 30}$)]
  } else if gid == "lagr" {
    [Lagr ($epsilon = 0.01$)]
  } else if gid.starts-with("baseline_p") {
    [Baseline ($p = #gid.slice(10)$)]
  } else if gid.starts-with("lagr_epsilon") {
    [Lagr ($epsilon = #gid.slice(12)$)]
  } else { raw(gid) }
}

#figure(
  table(
    columns: (auto,) + group-ids.map(_ => 1fr),
    align: (left,) + group-ids.map(_ => center),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header(
      table.cell(align: left)[*Metric*],
      ..group-ids.map(gid => table.cell(align: center)[*#pretty-label(gid)*]),
    ),
    table.hline(stroke: 0.6pt),
    ..rows.map(row => {
      let spec = display.at(row.metric)
      let lower-better = int(row.lower_is_better) == 1
      let means = group-ids.map(gid => float(row.at(gid + "_mean")))
      let best = if lower-better { calc.min(..means) } else { calc.max(..means) }
      (
        spec.label,
        ..group-ids.map(gid => {
          let cell = fmt-cell(row.at(gid + "_mean"), row.at(gid + "_std"), spec)
          if float(row.at(gid + "_mean")) == best { strong(cell) } else { cell }
        }),
      )
    }).flatten(),
    table.hline(stroke: 1pt),
  ),
  kind: table,
  caption: figure.caption(position: bottom)[
    Pendulum E1, clean-reward evaluation (200 episodes/seed). Baseline pools
    $p in {2, 10, 30}$ (6 seeds each, $n=18$); Lagrangian is $epsilon = 0.01$
    (6 seeds, $n=6$). Bold marks the better mean per row; see @sec-metrics for
    the zone depth-weighted metric.
  ],
) <tab-pendulum-pooled>

Lagrangian weighting matches or beats the pooled baseline on every metric in
@tab-pendulum-pooled: $4.2 times$ shallower zone penetration, higher return,
perfect and far more consistent upright success, and faster swing-up, at
equal wall-clock cost. @fig-alpha-trajectory shows why: $alpha$ rises while
the constraint is violated and decays toward 0 once $C < epsilon$, auto-tuning
the penalty instead of relying on a hand-picked $p$.

#figure(
  align(center)[#image("../figures/alpha_trajectory.png", width: 100%)],
  caption: [
    Lagrangian dual variable $alpha$ (left) and constraint value $C$ (right) over
    training, lagr $epsilon = 0.01$, pendulum E1, 6 seeds (thin lines), mean $plus.minus$
    std shaded. $alpha$ rises while $C > epsilon$, then decays toward 0 once the
    constraint is satisfied -- the auto-tuning behavior a fixed-weight baseline
    can't replicate.
  ],
) <fig-alpha-trajectory>

Early in training the policy has not yet learned to avoid the excluded band,
so the mean constraint value $C$ spikes to roughly $0.16$ -- sixteen times the
target $epsilon = 0.01$ -- around step 5k, while the underlying random-ish
exploration policy still routes through the zone. $alpha$ climbs in response,
from its initialized value of $5$ to a peak of about $12.5$ around step
30k-40k, sharply raising the effective penalty until the policy learns a
zone-avoiding route and $C$ falls back under $epsilon$. From there the two
signals decouple by seed: once $C$ stays below $epsilon$, dual ascent pulls
$alpha$ back down, and in 4 of 6 seeds it reaches exactly $0$ by the end of
training -- the penalty switches itself off entirely once it is no longer
needed. The remaining two seeds (final $alpha = 6.08$ and $16.50$) keep a
small residual weight, consistent with occasional late-training excursions
visible as the noisy individual $C$ traces that briefly poke back above
$epsilon$ in the right panel. No baseline weight is adjusted this way: a fixed
$p$ pays the same cost throughout training regardless of whether the
constraint is already satisfied.


The four panels in @fig-vis-trajectories trace the same route-around-vs-through trade-off the left/right
path split quantifies. Under $p=2$ the density visibly overlaps the red wedge
on both sides, and the swing direction is nearly a coin flip across seeds
(left $78.8%$, right $21.2%$) -- the penalty is too weak to consistently steer
the policy away from the shorter, zone-crossing path. $p=10$ mostly commits to
the longer right-hand route ($83.3%$ right) with only faint density inside the
wedge, and $p=30$ and lagr both route right on every seed ($100%$), leaving
almost no visible mass in the zone. The qualitative difference between $p=10$
and lagr that @tab-pendulum-pooled quantifies -- lagr's zone entries being
shallower, not just similarly rare -- is visible here too: where $p=10$'s
faint density inside the wedge reaches noticeably toward its center, lagr's
entries stay hugging the boundary, barely crossing the red line before turning
back.

#figure(
  align(center)[#image("../figures/vis_trajectories.png", width: 110%)],
  caption: [
    Visited states over the unit circle ($sin theta$, $cos theta$), hexbin density
    across all 6 seeds per group, pendulum E1. Red wedge marks the excluded zone.
    Baseline $p=2$ cuts through the zone; $p=10$/$p=30$ and lagr route around it,
    with lagr's path visibly hugging the boundary more tightly than $p=10$'s.
  ],
) <fig-vis-trajectories>



- LunarLander: same table and plots, framed as a generalization check -- does the pendulum
  ranking hold outside that setting?
#pagebreak()
#set page(columns: 1)
= Appendix <appendix>
== Pendulum: Lagrangian against each individual baseline setting <appendix-pendulum-by-p>

@tab-pendulum-pooled in @sec-results pools $p in {2, 10, 30}$ into a single baseline column.
@tab-pendulum-by-p below breaks that pool back out into its three individual settings, so lagr
can be read against each hand-tuned value on its own rather than against their average.

#let by-p-rows = csv("data/group_comparison_by_p.csv", row-type: dictionary)

#let by-p-display = (
  "Zone depth-weighted step fraction": (label: [Zone depth-weighted \ step fraction], percent: true, decimals: 3),
  "True return": (label: [True return], percent: false, decimals: 1),
  "Upright success rate": (label: [Upright success rate], percent: true, decimals: 1),
  "Time to upright (steps)": (label: [Time to upright (steps)], percent: false, decimals: 1),
  "Wall-clock training time (s)": (label: [Wall-clock training time (s)], percent: false, decimals: 1),
)

#let by-p-fmt-cell(mean, std, spec) = {
  let scale = if spec.percent { 100 } else { 1 }
  let unit = if spec.percent { "%" } else { "" }
  let m = calc.round(float(mean) * scale, digits: spec.decimals)
  let s = calc.round(float(std) * scale, digits: spec.decimals)
  [#m $plus.minus$ #s#unit]
}

#let by-p-group-ids = by-p-rows.first().keys().filter(k => k.ends-with("_mean")).map(k => k.slice(0, k.len() - 5))

#let by-p-pretty-label(gid) = {
  if gid.starts-with("baseline_p") {
    [Baseline ($p = #gid.slice(10)$)]
  } else if gid.starts-with("lagr_epsilon") {
    [Lagr ($epsilon = #gid.slice(12)$)]
  } else { raw(gid) }
}

#figure(
  block(width: 100%)[
    #set text(size: 8pt)
    #table(
      columns: (auto,) + by-p-group-ids.map(_ => 1fr),
      align: (left,) + by-p-group-ids.map(_ => center),
      stroke: none,
      table.hline(stroke: 1pt),
      table.header(
        table.cell(align: left)[*Metric*],
        ..by-p-group-ids.map(gid => table.cell(align: center)[*#by-p-pretty-label(gid)*]),
      ),
      table.hline(stroke: 0.6pt),
      ..by-p-rows.map(row => {
        let spec = by-p-display.at(row.metric)
        let lower-better = int(row.lower_is_better) == 1
        let means = by-p-group-ids.map(gid => float(row.at(gid + "_mean")))
        let best = if lower-better { calc.min(..means) } else { calc.max(..means) }
        (
          spec.label,
          ..by-p-group-ids.map(gid => {
            let cell = by-p-fmt-cell(row.at(gid + "_mean"), row.at(gid + "_std"), spec)
            if float(row.at(gid + "_mean")) == best { strong(cell) } else { cell }
          }),
        )
      }).flatten(),
      table.hline(stroke: 1pt),
    )
  ],
  kind: table,
  caption: figure.caption(position: bottom)[
    Pendulum E1, clean-reward evaluation (200 episodes/seed, 6 seeds per column, $n=6$
    throughout). Same metrics and evaluation protocol as @tab-pendulum-pooled, with the
    baseline broken out by individual $p$ instead of pooled. Bold marks the better mean per
    row across all four columns.
  ],
) <tab-pendulum-by-p>
  
== LunarLander: Lagrangian against each individual baseline setting <appendix-ll-by-p>
 
@tab-ll-pooled in @sec-results pools $p in {2, 10, 30}$ into a single baseline column.
@tab-ll-by-p below breaks that pool back out into its three individual settings, so lagr
can be read against each hand-tuned value on its own rather than against their average.
 
#let ll-by-p-rows = csv("data/ll_group_comparison_by_p.csv", row-type: dictionary)
 
#let ll-by-p-display = (
  "Zone depth-weighted step fraction": (label: [Zone depth-weighted \ step fraction], percent: true, decimals: 3),
  "True return": (label: [True return], percent: false, decimals: 1),
  "Landing success rate": (label: [Landing success rate], percent: true, decimals: 1),
  "Strict landing rate": (label: [Strict landing rate], percent: true, decimals: 1),
  "Crash rate": (label: [Crash rate], percent: true, decimals: 1),
  "Timeout rate": (label: [Timeout rate], percent: true, decimals: 1),
  "Wall-clock training time (s)": (label: [Wall-clock training time (s)], percent: false, decimals: 1),
)
 
#let ll-by-p-fmt-cell(mean, std, spec) = {
  let scale = if spec.percent { 100 } else { 1 }
  let unit = if spec.percent { "%" } else { "" }
  let m = calc.round(float(mean) * scale, digits: spec.decimals)
  let s = calc.round(float(std) * scale, digits: spec.decimals)
  [#m $plus.minus$ #s#unit]
}
 
#let ll-by-p-group-ids = ll-by-p-rows.first().keys().filter(k => k.ends-with("_mean")).map(k => k.slice(0, k.len() - 5))
 
#let ll-by-p-pretty-label(gid) = {
  if gid.starts-with("baseline_p") {
    [Baseline ($p = #gid.slice(10)$)]
  } else if gid.starts-with("lagr_epsilon") {
    [Lagr ($epsilon = #gid.slice(12)$)]
  } else { raw(gid) }
}
 
#figure(
  block(width: 100%)[
    #set text(size: 8pt)
    #table(
      columns: (auto,) + ll-by-p-group-ids.map(_ => 1fr),
      align: (left,) + ll-by-p-group-ids.map(_ => center),
      stroke: none,
      table.hline(stroke: 1pt),
      table.header(
        table.cell(align: left)[*Metric*],
        ..ll-by-p-group-ids.map(gid => table.cell(align: center)[*#ll-by-p-pretty-label(gid)*]),
      ),
      table.hline(stroke: 0.6pt),
      ..ll-by-p-rows.map(row => {
        let spec = ll-by-p-display.at(row.metric)
        let lower-better = int(row.lower_is_better) == 1
        let means = ll-by-p-group-ids.map(gid => float(row.at(gid + "_mean")))
        let best = if lower-better { calc.min(..means) } else { calc.max(..means) }
        (
          spec.label,
          ..ll-by-p-group-ids.map(gid => {
            let cell = ll-by-p-fmt-cell(row.at(gid + "_mean"), row.at(gid + "_std"), spec)
            if float(row.at(gid + "_mean")) == best { strong(cell) } else { cell }
          }),
        )
      }).flatten(),
      table.hline(stroke: 1pt),
    )
  ],
  kind: table,
  caption: figure.caption(position: bottom)[
    LunarLander, clean-reward evaluation (100 episodes/seed, 6 seeds per column, $n=6$
    throughout). Same metrics and evaluation protocol as @tab-ll-pooled, with the baseline
    broken out by individual $p$ instead of pooled. Bold marks the better mean per row across
    all four columns.
  ],
) <tab-ll-by-p>
 
== LunarLander trajectory comparison <appendix-ll-traj>
 
The appendix figure below aggregates the final evaluation trajectories of the LunarLander
runs for the baseline settings $p in {2, 10, 30}$ and the Lagrangian setting
$epsilon = 0.01$. All panels use the same $x$/$y$ axis limits so the trajectory density can
be compared directly across methods without differences caused by panel-specific scaling.
 
#figure(
  image("../figures/ll_combined_traj.png", width: 100%),
  caption: [
    Combined LunarLander trajectory comparison from the v3-matrix runs in the
    `runs/LL` directory. Each panel pools the final trajectory states from six seeds for a
    given setting; the red rectangle marks the excluded low-density box region in position
    space. Panels share the same axis scale and area to enable direct visual comparison of
    how each method navigates around the sparse region.
  ],
) <fig-ll-combined-traj>
  
== Initial exploratory experiment <appendix-pilot-experiment>

The following tables summarize the setup and results of the initial exploratory
experiment referenced in the methodology.

#let pilot-setup = csv("data/appendix_pilot_setup.csv")

#figure(
  block(width: 100%)[
    #set text(size: 7pt)
    #table(
      columns: (1.25fr, 1fr, 1fr, 1fr, 1.1fr, 1.35fr),
      align: left,
      inset: 4pt,
      stroke: none,
      table.hline(stroke: 1pt),
      table.header(..pilot-setup.first().map(cell => table.cell(align: left)[*#cell*])),
      table.hline(stroke: 0.6pt),
      ..pilot-setup.slice(1).flatten(),
      table.hline(stroke: 1pt),
    )
  ],
  caption: [Setup of the exploratory pilot experiment.],
) <table-pilot-setup>

// metrics as rows, methods as columns, best value per row in bold --
// same layout and convention as @tab-pendulum-pooled.
#let pilot-rows = csv("data/appendix_pilot_results.csv", row-type: dictionary)
#let clean-method(m) = m.replace(" (WINNER)", "")
#let pilot-methods = pilot-rows.map(r => r.at("METHOD"))

#let pilot-metrics = (
  (key: "ZONE-VISIT ↓", label: [Zone visit rate], lower-better: true, decimals: 3),
  (key: "RETURN ↑", label: [Return], lower-better: false, decimals: 1),
  (key: "ROBUSTNESS ↑", label: [Robustness], lower-better: false, decimals: 3),
  (key: "WEIGHTED TOTAL", label: [Weighted total], lower-better: false, decimals: 3),
)

#figure(
  block(width: 100%)[
    #set text(size: 7pt)
    #table(
      columns: (1.3fr,) + pilot-methods.map(_ => 1fr),
      align: (left,) + pilot-methods.map(_ => right),
      inset: 3pt,
      stroke: none,
      table.hline(stroke: 1pt),
      table.header(
        table.cell(align: left)[*Metric*],
        ..pilot-methods.map(m => table.cell(align: right)[*#clean-method(m)*]),
      ),
      table.hline(stroke: 0.6pt),
      ..pilot-metrics.map(spec => {
        let values = pilot-rows.map(r => float(r.at(spec.key)))
        let best = if spec.lower-better { calc.min(..values) } else { calc.max(..values) }
        (
          spec.label,
          ..values.map(v => {
            let cell = [#calc.round(v, digits: spec.decimals)]
            if v == best { strong(cell) } else { cell }
          }),
        )
      }).flatten(),
      table.hline(stroke: 1pt),
    )
  ],
  caption: [Results of the exploratory pilot experiment. Bold marks the better value per
    metric row; see @sec-metrics for definitions.],
) <table-pilot-results>

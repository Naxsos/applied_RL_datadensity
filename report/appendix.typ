#pagebreak()
#set page(columns: 1)
= Appendix <appendix>
== Initial exploratory experiment <appendix-pilot-experiment>

The following tables summarize the setup and results of the initial exploratory
experiment referenced in the methodology.

#let pilot-setup = csv("appendix_pilot_setup.csv")

#figure(
  block(width: 100%)[
    #set text(size: 7pt)
    #table(
      columns: (1.25fr, 1fr, 1fr, 1fr, 1.1fr, 1.35fr),
      inset: 4pt,
      stroke: 0.4pt,
      table.header(..pilot-setup.first().map(cell => strong(cell))),
      ..pilot-setup.slice(1).flatten(),
    )
  ],
  caption: [Setup of the exploratory pilot experiment.],
) <table-pilot-setup>

#let pilot-results = csv("appendix_pilot_results.csv")

#figure(
  block(width: 100%)[
    #set text(size: 7pt)
    #table(
      columns: (1.1fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
      align: (left, right, right, right, right),
      inset: 3pt,
      stroke: 0.4pt,
      table.header(..pilot-results.first().map(cell => strong(cell))),
      ..pilot-results.slice(1).flatten(),
    )
  ],
  caption: [Results of the exploratory pilot experiment.],
) <table-pilot-results>

# Reading the figures

Analysis reads locally generated runs; it does not call models. Run it on an
allocated CPU node when using a cluster. `docs/reproduction.md` gives the order
of operations. Final figures and seed-level CSVs go under ignored `outputs/`.

- **Finite-bandit reward and regret:** reward includes missing pulls as zero.
  Shared environment seeds and the common oracle make policies comparable.
  Cumulative curves show whether a policy finds good options sooner.
- **Completion and diversity:** separate failure to act from the options an
  agent selects. A copied arm beating a personal alternative is a conditional
  comparison, not the counterfactual value of another independent search.
- **Protected execution and independent search:** plot both raw solo/social
  performance and paired intervention effects. Fixing a mechanism need not
  produce a social advantage against an equally treated solo policy.
- **Skill-evolution learning curves:** whole-answer and individual-constraint
  accuracy on held-out tasks. Test feedback never enters learning. Average
  trajectory accuracy integrates the saved checkpoints.
- **Matched tokens:** compare social and solo at the smaller final expenditure
  of each seed pair, interpolating without extrapolation. Learning tokens count
  prompts and completions for every agent, including the discoverer's search.
  They exclude hidden test execution; end-to-end resource tables include it.
- **Lineages and copying:** revision–copy–revision ancestry is distinct from
  semantic skill diversity. Observation, first acquisition, deployment, and
  later revision are separate logged events. Peer order was fixed, so source
  preferences cannot be attributed solely to peer quality.
- **Copying-time interventions:** assigned early and distributed schedules
  both force four observations. Optional early access also changes frequency.
  These comparisons test schedules, not copying a fixed artifact at two times.

Aggregate agents within each population before reporting mean ± standard
error across seeds. The ten finite-bandit key tests use Bonferroni correction;
the three primary matched-token bandit tests form a separate family. Other
reported tests are nominal unless explicitly stated. Joint seed/task bootstrap
uncertainty is a separate sensitivity analysis, not additional independent seeds.

Historical original-loop comparisons are optional and require their own results;
they are never pooled with corrected-loop data. The release does not bundle
those runs or use hand-entered results in place of a new analysis.

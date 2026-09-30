"""Build the paper analyses from completed runs; no model calls."""
import runpy
from pathlib import Path
import plot_short_report as plot
import analyze_paper_revision as audit


def main():
    root = Path(__file__).resolve().parents[1]
    plot.setup()
    plot.PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    plot.paper_figures()
    plot.main_r1()
    plot.rung1_controls()
    plot.rung3_trajectory()
    plot.rung3_access()
    plot.rung3_deoe()
    plot.rung2_volatility()
    audit.primary_bandit_diagnostics()
    plot.paper_rung1_diagnosis_and_compact_intervention()
    runpy.run_path(str(root / 'analysis/analyze_bandit_matched_tokens.py'), run_name='__main__')
    plot.paper_skill_results()
    audit.resource_accounting()
    cost = audit.matched_cost()
    lineage, peers, budgets, items = audit.raw_audit()
    uncertainty = audit.bootstrap(items)
    audit.figures(cost, lineage, peers, uncertainty)
    audit.supplemental_audits()
    audit.timing_contrasts()
    audit.paired_significance()
    runpy.run_path(str(root / 'analysis/bandit_key_tests.py'), run_name='__main__')
    # This is the final prompt+completion-token comparison, not the historical
    # completion-only or USD audit. Run last so the primary plot uses tokens.
    runpy.run_path(str(root / 'analysis/analyze_matched_learning_tokens.py'), run_name='__main__')
    plot.paper_feedback_bandits()
    plot.paper_skill_origin_costs()
    plot.paper_observation_networks()
    plot.paper_copy_timing()
    plot.paper_offline_skill_check()
    plot.paper_intro_figure()
    print('Figures and seed-level summaries written to outputs/.')


if __name__ == '__main__':
    main()

# Fast homogeneous social-learning references

`code/social_baseline.py` implements **discounted explore–observe–exploit (DEOE)** for
rungs 1 and 2 and **hierarchical social UCB (H-SUCB)** for rung 1. All ten agents
within a population use the same policy and have independent random streams.
There are no designated innovators, privileged teachers, mixed-policy opponents,
model calls, or offline fitting. Parameters were fixed before each matched sweep,
not selected for a favorable comparison with its results.

## Hierarchical social UCB

H-SUCB adapts the two-stage agent-then-arm construction in Shin, Lee and Ok
(2022), while making peer selection an the experiment observation rather than
letting a central principal directly choose every agent's arm:
https://proceedings.mlr.press/v151/shin22a.html

Each agent maintains an outer Gaussian UCB over ten sources: itself and all nine
peers. If self is selected, an inner Gaussian UCB chooses among the 25 task arms.
If peer `j` is selected, the agent explicitly observes `j`'s latest pull strictly
before the current round. The observed payoff updates `j`'s outer index and the
observed arm's lower-level estimate. The lower UCB then chooses the scored arm;
selecting a person acquires evidence and does not force blind imitation. If self
is selected, the lower UCB's realized pull updates self's outer index. Every agent
makes exactly one scored pull per round.

The reported `hierarchical_payoff` mode requires action-plus-payoff observation.
The peer's realized payoff is added to the lower arm UCB with weight one before
execution. Unit weight is justified here because agents face the same arm
distributions and reward noise is independent by agent and round. It should not
be carried to heterogeneous-agent tasks without a transfer model. Action-only
hierarchical source selection is intentionally omitted: an arm identifier alone
does not provide a principled reward sample for either UCB level.

All outer sources have one pseudo-observation at the known reservoir mean. Self's
prior mean is higher by `0.5 * reward_noise_std`; this is a mild, explicit initial
preference for independent search whose influence vanishes with observations.
Both UCB levels use exploration coefficient `sqrt(2)`. At disclosed payoff-regime
boundaries, source and arm reward statistics reset exactly as in the existing
solo-UCB reference; peers are unavailable as sources until they have acted in the
new regime. First-acquisition provenance remains intact across regimes.

H-SUCB is an algorithmic social-selection baseline, not a randomized causal
control. Sampling different people does not ensure different candidates when
those people have converged on the same arm. The separately randomized forced
independent-search intervention remains the test of whether restoring innovation
supply changes diversity and reward.

## Relationship to the tournament winner

Rendell et al. (2010), *Why Copy Others?*, describes the winning `discountmachine`
as discounting information with age and relying heavily on observation:
https://lalandlab.wp.st-andrews.ac.uk/files/2015/08/rendell_et-al_2010.pdf

Carr's dissertation, section 6.1 / Algorithm 3, documents its pretrained decision
network, bootstrap innovation, and periodic observation heuristic:
https://api.drum.lib.umd.edu/server/api/core/bitstreams/96969709-116c-4c6d-93c2-7f587d6a1c11/content

DEOE is an inspired reference, **not a reproduction of discountmachine**. It
retains age discounting and periodic checks but replaces the pretrained network
with explicit rules. Continued independent search is a deliberate adaptation
for a homogeneous, finite-horizon population without demographic replacement.
We do not claim it is optimal or that it must beat solo learning.

## Frozen DEOE policy

1. Start by independently acquiring an option. Subsequently attempt independent
   discovery with probability `1/sqrt(round+1)`. Finite agents choose uniformly
   among personally unknown arms; infinite agents sample a uniform coordinate.
   This baseline does not perform a specialized local search on structured tasks.
2. If not independently discovering, a social agent queries one uniformly chosen
   peer when 20 rounds have elapsed since its last query, after a surprising
   payoff, or with probability `1/sqrt(round+1)`. A solo agent skips this step.
   Peer choice uses no knowledge of unqueried peers' rewards.
3. Maintain exponentially weighted noisy samples, decay 0.95 per round. Discount
   each arm's last estimated value toward the average of the agent's arm estimates
   as it ages. A sample more than three noise-adjusted standard deviations from
   its previous estimate resets that arm's estimate and triggers a peer check.
   The reward-noise scale is a supplied task parameter; shock dates, true means,
   the landscape function and the optimal arm are unavailable to the policy.
4. Pull the arm with the best discounted estimate. Personally untested arms seen
   in action-only observations receive a trial before exploitation. Rung-1
   independent discovery itself is the scored pull. Rung-2 innovation is an
   unscored probe, followed by a scored pull from the updated repertoire.

Observation is limited to one explicit query per agent-round, using the existing
environment's frozen pre-round snapshot. Payoff-visible queries supply a noisy
realized reward, not a true mean. Old observations retain their original dates;
repeated queries of the same source event cannot count as independent samples.
First-acquisition labels remain permanent. All agents earn at most one scored
pull per round. The policy can observe or innovate and then pull in the same
round, as the existing LLM action interface permits.

## Matching and interpretation

The sweep extracts all unique environment configurations from completed V3
LLM runs, using the original environment implementations, ten agents, 100 rounds,
and the same eight seeds. It runs solo, action-only social, and payoff-visible
social populations for each environment. Random streams for independent-search
decisions are shared between policy modes; available options can diverge after
social acquisition. The condition mapping explicitly reuses a single reference
across model/prompt/budget conditions with the same environment; those reused
references are not independent replications.

Like the existing UCB reference, DEOE uses zero LLM completion tokens and does
not simulate language-model reasoning failures. It is **not token-matched** and
its zero-token reward efficiency should not be plotted as a finite number.
Reward-versus-round measures attainable algorithmic task performance; social
minus solo within DEOE measures the benefit of observation for this rule. A gap
between DEOE and an LLM combines policy and execution differences and is not a
causal estimate of an LLM's lost social-learning value. Carry-over and prompt
guidance do not change this algorithm. Short CPU runtime is recorded per run.

Outputs contain seed-level reward, conditional reward, completion, discovery,
social-first use/quality, observation frequency, diversity and learning curves.
Uncertainty and contrasts are computed across eight matched populations, not
across individual agents or rounds. Original LLM and UCB results are preserved.

# Why lean-002 is plateauing: comparison with established poker systems

This review compares the active lean implementation with primary research and
OpenSpiel source. It separates confirmed code properties from causal hypotheses.
No further training, checkpoint migration or live deployment was performed.

## Finding 1: betting history is erased (highest-priority structural defect)

`poker/lean_abstraction.py:64` uses position, active-player counts, cards, coarse
price/pot/SPR and action count. It does not encode the betting sequence, aggressor,
prior street actions or the identities of all remaining players. It also merges
4/5/6 active players postflop. `observation_key` discards observation history.
Training calls `FastGame.act(..., False)`, which does not append history at all.
Adding history only in the live adapter would therefore mismatch training.

The reproducible audit found different legal paths from the same deal with
identical hole/board cards and legal menus mapping to the same key. Example:
hero 9d9c, key `0241|77o|150|3f`, legal call/fold/200. One path has raises
5/13/23/106 and reaches pot342/call83; another has raises4/7/17/35/68/132 and
reaches pot368/call97. Values are engine half-BB units. Both must use the exact
same action probabilities. The issue is broader than these extreme examples:
even an identical present pot cannot encode what previous bets revealed.

Evidence: `runs/bot/architecture-audit-001/state-loss.json`.
Reproduction: `python -m scripts.lean_state_audit --out NEW_FILE.json`.
A test replays both paths through legal actions and verifies the collisions.
This proves information loss; it does not measure its payoff cost.

Pluribus keys decisions by abstract action sequence plus a card-information
bucket. Its blueprint card abstraction uses 169 preflop classes and 200 buckets
per later street. It refines play with range-based depth-limited search and handles
off-tree bets by adding actions and resolving. Its postflop blueprint averages
current-strategy snapshots, unlike our accumulation at all sampled opponent nodes.
Source: [Pluribus supplementary material, pp.12–15 and 21–22](https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf).

## Finding 2: hand features confuse present rank with strategic value

`card_features` ranks the current hand against board-compatible two-card holdings,
including holdings overlapping the hero's cards. It does not remove those blockers.
On flop/turn it does not sample future cards. Thus its value is neither showdown
equity nor equity against a betting-conditioned range. Draw flags record only one
of flush/straight, and lose draw quality, redraws, blockers and much board structure.
Even perfect lookup coverage can mean a well-populated compromise between hands
that should play differently. Expanding bucket counts alone is not the solution.

The audit includes a board royal flush where exact heads-up equity is 0.5 and the
feature strength bucket is 3. This is not an evaluator bug: percentile rank and
showdown equity are different quantities. It illustrates why callers must not
interpret the current feature as probability of winning.

DeepStack maintains range/value information during continual resolving; its leaf
values depend on ranges, not just the hero's present rank. It is a heads-up system,
so its guarantees do not transfer to this six-player game.
Source: [DeepStack paper, authors' copy](https://johanson.ca/publications/poker/2017-science-deepstack/2017-science-deepstack.pdf).
Deep CFR offers learned function approximation as an alternative to manual
abstraction; this is a larger engineering change, not an automatic fix.
Source: [Deep CFR, ICML 2019](https://proceedings.mlr.press/v97/brown19b.html).

## Finding 3: blueprint-only play and a restrictive action model

`LeanPolicy.act` samples a row or invokes the tight rule fallback. It has no
range-aware search to recover detail discarded by the blueprint. `FastGame.menu`
provides half-pot/pot raises and only near-pot all-ins for learner seats. It cannot
represent many small/large bets, and learners never face many live bet sizes in
self-play. The gate does let opponents use the full menu, which reveals this
mismatch. No inference that simply removing the menu restrictions fixes learning
is warranted: the earlier full menu also produced bad behavior and large trees.

Pluribus's combination of blueprint and search is the relevant inspiration.
Neither neural networks nor an opponent-specific adaptation module is a prerequisite
for a strong blueprint. Range reasoning in search differs from modeling an
opponent's personal tendencies. A history-conditioned blueprint can implicitly
encode range effects without explicitly updating a range tracker at runtime.
Source: [Pluribus main paper](https://noambrown.com/papers/19-Science-Superhuman.pdf).

## Finding 4: averaging differs, but full averaging is not a proven cure

Our solver updates averages at EVERY non-traverser learned node. OpenSpiel's
simple estimator updates only the designated next player on each traverser's pass;
it also implements full averaging separately. Its comments acknowledge multiplayer
bias/support concerns yet explicitly retain simple averaging as default for speed
and empirical performance. Therefore the earlier suggestion that lack of full
averaging explains failure was too strong. A controlled estimator comparison is
needed. Multiplying update mass by iteration weight does not resolve state aliasing.
Source: [OpenSpiel external_sampling_mccfr.py, lines30–46 and146–152](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/external_sampling_mccfr.py).

## Finding 5: the evidence process rewarded the wrong milestones

The original gates allowed strategies that lost heavily to fixed rules. The new
gate is better, but tuning to those same four styles can still overfit. Lookup
coverage, throughput, and positive-regret rows are not playing-strength measures.
The mixture pilot improved exposure and raw station play; it did not beat the
rule-only control. Those results support a distribution diagnosis, not a complete
architecture. Compare against fallback alone and unseen opponents on frozen deals;
keep independent training seeds separate from two continuations of one parent.

## Recommended implementation order

1. Freeze lean-002 and the mixture pilot as baselines. Build a small known-game
   adapter and compare traversal updates and policy quality with an established
   implementation, including the chosen averaging estimator.
2. Create a new abstraction version retaining abstract betting history and a
   consistent action-slot meaning, with training/inference parity. Start with a
   bounded game (limited streets/stacks/actions) so history does not explode tables.
   Preserve distinct aggressors and active-player identities where meaningful.
3. For the first poker proof of improvement, solve a small river subgame with
   explicit ranges and exact cards. Show advantage against independent opponents
   and quantify exploitability only where a suitable two-player setting permits it.
4. Add potential-aware card features/buckets, then extend the validated blueprint.
   Evaluate history and card changes separately to attribute improvements.
5. Add range-aware limited-depth search and principled off-tree handling once the
   small-game solver works. Avoid perfect-information rollouts that know opponents'
   actual cards or choose a different policy separately for each hidden world.
6. Use bounded OpenPoker sessions for discovery of unsupported states after local
   correctness checks. Retain decisions, coverage and hand results. Live winnings
   alone cannot isolate architecture changes or validate multiplayer equilibrium.

The next concrete work should be the solver/representation baseline above, rather
than another long mixture continuation. The strongest code-backed concern is
strategic information erased from the key. The amount of loss attributable to each
finding remains unmeasured.

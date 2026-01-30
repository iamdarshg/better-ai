TODO

- Phase 7: Real GBNF library and tests
  - Implement or integrate a proper GBNF parser library to replace the ad-hoc grammar constraints in better_ai/models/features/gbnf_constraint.py.
  - Options::
    - Integrate an existing GBNF parser library (e.g., a Python-based GBNF/grammar tool) and expose a small API to validate sequences against a GBNF grammar.
    - Or implement a minimal in-house GBNF parser covering the required grammar subset used by the project.
  - Add unit tests for the GBNF path to ensure syntactically valid sequences pass and invalid ones are blocked.

- Tests to add/mature mocks (Phase 7 +)
  - better_ai/models/tot.py: Add tests with mocks on lines 23 and 27 (generate_thoughts and evaluate_states) – done in tests/unit/test_tot.py (and enhanced as needed).
  - better_ai/training/curriculum_mcts_trainer.py: Add tests that mock internal flows at lines 264 and 293 (MCTS integration and GRPO step path) to verify orchestration without heavy components.
  - better_ai/training/arpo.py: Add tests mocking the entropy-based rollout paths at line 404 to exercise ARPO flow without full model runs.
  - better_ai/data/dataset_config.py: Add tests around loading datasets and stage filtering using mocks; test edge cases and error handling (line 17 vicinity).

- Implementation notes
  - Ensure tests are isolated and avoid long-running heavy model training.
 Provide clear, deterministic mocks to validate control flow and data paths.

- Deliverables
  - ROOT-level TODO.md updated with Phase 7 goals and a plan for tests.

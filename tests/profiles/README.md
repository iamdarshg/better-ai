Flame Graphs and Resource Tagging

- Resource tagging:
  - Mark tests with @high_resource to include them in automated profiling.
  - Mark tests with @low_resource to separate profiling if needed later.

- Flame graphs:
  - Generated per high-resource test as SVG files under:
    tests/profiles/high_resource/<sanitized_test_id>.svg
  - Low-resource flame graphs will live under:
    tests/profiles/low_resource/<sanitized_test_id>.svg

- Reading graphs:
  - Use flamegraph viewers or Chrome tracing to inspect hotspots.

- Extending tagging:
  - Add @high_resource or @low_resource to new tests to include them automatically.

This README is a lightweight guide. More details are in the CI workflow and profiling scripts.

# Day 26

**Topic:** Observability: Metrics, Tracing & Dashboards
**Date:** 2026-04-06
**Layer:** Tooling

## What I explored
Built a Prometheus-style metrics registry for an inference server, implemented distributed tracing for a full request lifecycle, and designed a Grafana dashboard with alerting thresholds for all key inference metrics.

## Key insight
Distributed traces reveal that queue wait time — not actual prefill compute — often dominates TTFT during traffic spikes. Without tracing, you'd optimize the wrong thing.

## Code / experiment
Notebook: [`observability-metrics-tracing.ipynb`](./observability-metrics-tracing.ipynb)
Key demo: Grafana dashboard simulation + distributed trace waterfall visualization

## References
- *Inference Engineering* Ch 7.4.3 (Philip Kiely, Baseten Books 2026)
- Prometheus documentation
- OpenTelemetry specification

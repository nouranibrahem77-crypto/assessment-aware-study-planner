# Monotonicity Experiment

- Model family: `neural_network`
- Model identity: `040c2af6327e413c71769c29f5fd27802dba65bd18d970d92e68930af70c7679`
- Model digest: `ab6d3a07b9fc4768e2e5341583f9a8e49c8e03e0520b26cf3daa9d7372e03e9a`
- Profile manifest SHA-256: `ac6f645af10d5d6ed4085ec2e1946bda7bfa87b7266144592e28941a70f4373f`
- Feature CSV SHA-256: `40aaa5f029e0c002b75e06e0f009fe315d5765c83d01e621604bfc52b61d280f`
- Click mapping: `low:content_navigation:22`
- Maximum total sessions per assignment: 2
- Seeds: 42, 123, 2025, 31415, 271828

A violation is a one-session allocation increase that lowers the predicted mark. Only predictions whose marginal mapping support is in range are included in comparisons. Out-of-range and unassessed allocations are reported separately.

| Seed | Profile runs | Comparisons | Violations | Pooled rate | Mean profile rate | Profile-rate SD | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 30 | 1800 | 480 | 0.2667 | 0.2667 | 0.1721 | 0 |
| 123 | 30 | 1800 | 480 | 0.2667 | 0.2667 | 0.1721 | 0 |
| 2025 | 30 | 1800 | 480 | 0.2667 | 0.2667 | 0.1721 | 0 |
| 31415 | 30 | 1800 | 480 | 0.2667 | 0.2667 | 0.1721 | 0 |
| 271828 | 30 | 1800 | 480 | 0.2667 | 0.2667 | 0.1721 | 0 |
| overall | 150 | 9000 | 2400 | 0.2667 | 0.2667 | 0.1698 | 0 |

Predictions under synthetic session-to-click perturbations describe model behavior and do not establish a causal effect of study time.

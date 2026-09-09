# Security

Treat CUDA performance PRs as untrusted code. Do not execute arbitrary fork PRs on persistent GPU hosts containing SSH keys, cloud credentials, production model credentials, writable shared data, or long-lived GitHub tokens.

Use disposable GPU runners for authoritative evaluation.

---
status: accepted
date: 2026-11-14
---

# Verda as cloud provider

ChatChTG is a Junction demo, so we need infrastructure that is cheap and fast to set up rather than feature-rich. We host the project on Verda: it is a Junction partner whose credits cover our costs, and its plain VM offering needs very little setup. This decision covers the project's own infrastructure only; model inference is not tied to Verda and may be consumed from other providers.

## Consequences

- Costs depend on the Junction credits. If the project outlives the demo, the provider choice should be revisited.
- Code that calls inference must not assume the model runs on Verda, or on the same host as the application.

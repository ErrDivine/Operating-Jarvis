# Guide on dataset generation

## Recommended recipe (works well for small models like Qwen3-4B)
### Core coverage (50–60%). 
- Every tool at least N times (N≈50–200 depending on importance).  
- Each required slot covered with all enumerated values at least a few times.    
- Minimal natural-language variety per intent (5–10 paraphrases).    
### Confusers & counterfactuals (20–30%)
- Pairs that your agent confuses (e.g., SwitchApp vs ShutDown, connect vs disconnect, “昨天/最近7天/最近30天”).
- Near-miss ASK vs CALL cases (when a required slot is missing → ask, not call).
- Confirmation-required/destructive actions (reboot/shutdown) with positive+negative examples.
### Failure replay (15–25%)
- Mine recent mistakes from eval/logs, dedup/normalize, and add them.
- Up-weight these samples (2–4×) or upsample them so the model learns the fix without forgetting the base.
### Format & robustness (5–10%)
- Strict formatting stress tests: wrong quotes, case, extra/unknown args, missing <tool> tag → teach the model to output the correct canonical form.

## Idea on the dataset
- Datase should include the core, fundamental situations that the agent basically needs. The dataset for each function should include data classified on the parameters; each parameter has several entries of training data exploring the occassions they are used. Then for usually misunderstood counterparts, add corresponding entry pairs to enhance the understanding. 
- The dataset should be strictly following our preset router-planner structure. One for router, 24 for planner.
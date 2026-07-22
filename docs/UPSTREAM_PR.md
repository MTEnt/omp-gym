# Upstream plan (OMP)

## What we will PR first

- Optional: document external gym tools that read OMP sessions  
- Prefer **extension + skill path** over monorepo bulk  

Suggested OMP PR series:

1. **Docs**: session JSONL location + skill discovery dirs (if not already clear)  
2. **Extension example**: `/gym` thin wrapper calling external `omp-gym`  
3. **Optional schema**: stable `omp session export --gym` for harvesters  

## What stays out of OMP core (initially)

- Replay/reflect optimizer  
- launchd/cron scheduler  
- Skill edit gate  

Those live in `MTEnt/omp-gym` until proven.

## Compatibility

- Read-only session harvest  
- No requirement to patch OMP for v0.1  
- Extension uses public `registerCommand` API  

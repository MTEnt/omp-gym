# omp-gym design

## Goal

Give OMP an overnight **skill gym**: improve natural-language skills from real usage, without forking OMP or fine-tuning weights.

## Non-goals (v0.1)

- Full SkillOpt paper benchmarks / WebUI  
- Automatic adopt without user action  
- In-tree OMP fork  

## Loop

1. **Harvest** OMP session JSONL (`~/.omp/agent/sessions/**/*.jsonl`)  
2. **Mine** recurring user intents → tasks  
3. **Replay** tasks with current skill (backend: mock → omp → API)  
4. **Reflect** optimizer proposes bounded markdown edits  
5. **Validate** held-out gate; accept only strict improvement  
6. **Stage** proposal under `.omp/gym/proposals/`  
7. **Adopt** user applies to target `SKILL.md` with backup  

## Trust boundaries

- Harvest is local read-only  
- Redact secret-shaped strings best-effort  
- Real backends will send task text to a model — document clearly  
- Mock backend never mutates skills  

## OMP integration

- Binary: `omp-gym`  
- Extension: `extensions/omp/gym.ts` → `/gym`  
- Skills remain plain markdown OMP already loads  

## v0.2 targets

- `omp -p` replay runner  
- OpenAI-compatible reflect  
- Unified diff stage + adopt with `.bak`  
- Linux systemd user timer parity  

# Claude Code skills used in this repo

Two third-party skill packs are installed project-locally in `.claude/skills/`
(not committed; recreate with `bash scripts/sync_skills.sh`). Both are MIT licensed.

| Pack | Source | Pinned commit |
|---|---|---|
| superpowers (Jesse Vincent) | https://github.com/obra/superpowers | `bb92a77741419a4ab5f06e711a283343f1ada0c3` (2026-10-09) |
| claude-skills (Alireza Rezvani) | https://github.com/alirezarezvani/claude-skills | `19392f7a08264ed00486a251f5b2098321771f94` (2026-08-26) |

The clones live in `third_party/` (gitignored) so the pins stay auditable. Only
the skills below were enabled; the packs hold 15 and 733 skills respectively and
loading all of them would flood every session's skill list.

## Enabled skills (21)

**From superpowers (12)** — process discipline: `brainstorming`, `writing-plans`,
`executing-plans`, `subagent-driven-development`, `dispatching-parallel-agents`,
`systematic-debugging`, `test-driven-development`, `verification-before-completion`,
`requesting-code-review`, `receiving-code-review`, `finishing-a-development-branch`,
`writing-skills`.

**From claude-skills (9)** — ML / data / review: `senior-ml-engineer`,
`senior-data-scientist`, `senior-data-engineer`, `tdd-guide`, `adversarial-reviewer`,
`code-reviewer`, `performance-profiler`, `spec-driven-workflow`, `tech-debt-tracker`.

## Deliberately not installed

- **superpowers `using-superpowers` skill and its SessionStart hook.** The hook
  injects a preamble into every session telling the model it must invoke a skill
  before *any* response, including clarifying questions. That conflicts with this
  repo's `.kiro/steering/credit-efficiency.md` (plan first, don't burn credits).
- **superpowers `using-git-worktrees`, `diagnosing-superpowers`.** Not needed here.
- **The other ~725 claude-skills skills** (marketing, finance, regulatory, C-level
  advisors, cloud, security...). Off-topic for an MS/MS structure-prediction project.

## Safety review done before enabling (2026-10-17)

Read-only scan, nothing executed. No `curl | sh`, base64-decode-and-run,
credential reads, or destructive deletes in any enabled `SKILL.md`. Hits in the
wider packs were documentation (security skills listing risk patterns, API-key
mentions in an agent launcher). Of the 35 helper scripts copied with the enabled
skills: `code-reviewer` shells out to `git`; `writing-skills/render-graphs.js`
calls `execFileSync` (a dev tool, never invoked by the skill text); the
network-calling files are sample fixture apps under `tech-debt-tracker/assets/`.
Skill text is treated as untrusted instructions: it can shape how Claude works,
but it does not override the user's instructions in `CLAUDE.md`, `.kiro/steering/`
or the chat.

## Where each skill earns its keep on this project

- `verification-before-completion` / `tdd-guide` / `test-driven-development`:
  every scoring or split change needs a failing test first. The two fusion bugs
  found so far (zero-weighted FPNet, raw dot-product weight) were exactly what
  these catch.
- `systematic-debugging`: the class-3 leak (stored vs scorer key) was found by
  hypothesis-driven tracing; this is the same loop.
- `adversarial-reviewer` / `code-reviewer`: use on `pipeline.py`, `split.py` and
  any new channel before a submission; leakage is the failure mode that costs
  private-LB rank.
- `senior-ml-engineer` / `senior-data-scientist`: drift monitoring, CV design,
  calibration for the reranker confidence gate (literature review §6, item 9).
- `writing-plans` / `executing-plans` / `subagent-driven-development`: the
  ICEBERG/GLACIER integration and contrastive retrieval head are multi-step.

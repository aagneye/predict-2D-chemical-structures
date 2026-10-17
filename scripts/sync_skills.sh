#!/usr/bin/env bash
# Re-create .claude/skills from two third-party skill repos at PINNED commits.
#
# Why a script instead of committing the skills: they are MIT-licensed third-party
# content (~2 MB, 21 skills, 35 helper scripts), so we pin exact commits and fetch
# them on demand rather than vendoring them into this repo's history. Skills are
# plain-text instructions; this script only copies files and NEVER executes
# anything from the clones. The superpowers SessionStart hook is deliberately not
# installed (it injects a "you must invoke skills" preamble into every session).
#
# Usage: bash scripts/sync_skills.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD="$ROOT/third_party"
SKILLS="$ROOT/.claude/skills"

SUPERPOWERS_URL=https://github.com/obra/superpowers.git
SUPERPOWERS_SHA=bb92a77741419a4ab5f06e711a283343f1ada0c3
CLAUDE_SKILLS_URL=https://github.com/alirezarezvani/claude-skills.git
CLAUDE_SKILLS_SHA=19392f7a08264ed00486a251f5b2098321771f94

# name -> path inside the repo
SUPERPOWERS_SKILLS=(
  brainstorming writing-plans executing-plans systematic-debugging
  test-driven-development verification-before-completion requesting-code-review
  receiving-code-review dispatching-parallel-agents finishing-a-development-branch
  subagent-driven-development writing-skills
)
CLAUDE_SKILLS_PATHS=(
  engineering-team/skills/senior-ml-engineer
  engineering-team/skills/senior-data-scientist
  engineering-team/skills/senior-data-engineer
  engineering-team/skills/tdd-guide
  engineering-team/skills/adversarial-reviewer
  engineering-team/skills/code-reviewer
  engineering/skills/performance-profiler
  engineering/skills/spec-driven-workflow
  engineering/skills/tech-debt-tracker
)

fetch_pinned() {  # url sha dir
  local url="$1" sha="$2" dir="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dir" fetch --quiet --unshallow || true
  git -C "$dir" checkout --quiet "$sha"
  echo "pinned $(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}

mkdir -p "$THIRD" "$SKILLS"
fetch_pinned "$SUPERPOWERS_URL" "$SUPERPOWERS_SHA" "$THIRD/superpowers"
fetch_pinned "$CLAUDE_SKILLS_URL" "$CLAUDE_SKILLS_SHA" "$THIRD/claude-skills"

copy_skill() {  # src_dir name
  local src="$1" name="$2"
  [ -f "$src/SKILL.md" ] || { echo "ERROR: $src has no SKILL.md" >&2; exit 1; }
  rm -rf "${SKILLS:?}/$name"
  cp -r "$src" "$SKILLS/$name"
}

for name in "${SUPERPOWERS_SKILLS[@]}"; do
  copy_skill "$THIRD/superpowers/skills/$name" "$name"
done
for path in "${CLAUDE_SKILLS_PATHS[@]}"; do
  copy_skill "$THIRD/claude-skills/$path" "$(basename "$path")"
done

echo "installed $(( ${#SUPERPOWERS_SKILLS[@]} + ${#CLAUDE_SKILLS_PATHS[@]} )) skills into $SKILLS"

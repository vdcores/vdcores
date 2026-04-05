# Task Records

This directory holds tracked task state for work that may span multiple conversations.

## When To Use

- Use one file here when the user names a task explicitly.
- Use one file here when the work is likely to continue across sessions.
- Skip this directory for tiny one-shot edits that do not need a handoff record.

## Naming

- Use a stable slug: `agents/tasks/<task-slug>.md`
- Prefer short lowercase names with hyphens, for example:
  - `agents/tasks/compiled-mode-cleanup.md`
  - `agents/tasks/qwen3-prefill-bringup.md`

## What To Record

Each task file should stay concise and include:

- title
- status
- created / updated dates
- short description
- current state
- recent progress
- remaining TODOs
- blockers / assumptions
- key files / commands / artifacts

## Relationship To `.agentlog/`

- Keep the durable task summary here.
- Put bulky command transcripts, temporary experiments, and machine-specific notes in `.agentlog/`.
- If a `.agentlog/` entry matters for resuming the task, link or summarize it here.

## Lifecycle

1. When starting a new named task, create a new task file from `agents/tasks/TEMPLATE.md`.
2. When continuing a task, read the existing task file first and update it during the same turn.
3. When the task is done, mark it complete and leave a short final outcome plus any follow-up work.

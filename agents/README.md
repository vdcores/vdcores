# Agent Notes

This directory holds the in-tree, high-signal context for agent work in `vdcores`.

## Layout

- `workflows/`: reusable procedures for future agent sessions.
- `knowledge/`: concise project notes captured from repository inspection.
- `tasks/`: tracked multi-conversation task records with progress, TODOs, blockers, and handoff state.

Detailed per-task logs live in the gitignored `.agentlog/` directory instead of this tree.

## Conventions

- Prefer short Markdown files with concrete file paths and commands.
- Keep `agents/` focused on durable summaries, reusable workflows, and stable repo knowledge.
- Keep `agents/tasks/` focused on resume-friendly task state rather than raw transcripts.
- Put one-off task history, full verification transcripts, and environment-specific blockers in `.agentlog/`.
- Update knowledge notes when you confirm new structural information that will likely matter again.
- When a task introduces an important repo change or durable lesson, persist a concise summary here in `agents/workflows/` or `agents/knowledge/`.
- When a task spans multiple conversations, keep one tracked file in `agents/tasks/<task-slug>.md` and update it as the task evolves.

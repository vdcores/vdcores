# Task Tracking Workflow

Use this when the user refers to named work that may continue across multiple conversations.

## Goal

Keep a small tracked task file under `agents/tasks/` so future sessions can resume from repository state rather than chat history.

## Start A Task

1. Choose a stable slug from the user phrasing.
   - Example: `compiled-mode-cleanup`
2. Create `agents/tasks/<slug>.md` from `agents/tasks/TEMPLATE.md`.
3. Fill in:
   - title
   - status
   - created / updated dates
   - short description
   - initial TODOs
   - known blockers / assumptions

## Continue A Task

1. Read `agents/tasks/<slug>.md` first.
2. Use that file as the authoritative summary of current task state.
3. Update the same file during the turn:
   - refresh `Updated`
   - append a short progress bullet
   - edit TODOs / blockers / next step
   - add new key files, commands, or artifacts if they matter for handoff

## Finish A Task

1. Mark `Status: done`.
2. Add a short final progress entry with the outcome.
3. Replace `Next Step` with follow-up work only if something intentionally remains.

## Scope Discipline

- Keep task files concise and handoff-oriented.
- Do not dump long command output into `agents/tasks/`.
- Put heavy logs in `.agentlog/` and summarize only the durable takeaway in the task file.
- If the work produces reusable guidance, also update `agents/workflows/` or `agents/knowledge/`.

## Matching User Language

- If the user says `start task X`, create the task if it does not exist.
- If the user says `continue task X`, prefer the existing matching slug and update it.
- If two task files could match, pause and resolve the ambiguity before changing both.

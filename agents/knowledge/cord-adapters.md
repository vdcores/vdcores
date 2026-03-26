# Cord Adapters

Schedule-side cord conversion now lives in wrapper objects instead of `SchedCopy(..., cords=...)`, `SchedRope(..., cords=...)`, or `SchedGemv(..., cordconv=...)`.

## Core Pattern

- `python/dae/tma_utils.py` defines `CordAdapter` wrappers around existing cordable objects.
- Schedules in `python/dae/schedule.py` keep their native `.cord(...)` call shapes and call `.cord(...)` directly on the wrapped inputs.
- The wrappers translate from schedule-space coordinates to the wrapped object's real coordinate shape.

## Current Adapters

- `StaticCordAdapter`: ignore schedule args and return the wrapped object unchanged
- `wrap_static(*tmas)`: convenience helper for building tuples of `StaticCordAdapter`
- `ToConvertedCordAdapter`: generic callable-based translation
- `ToLinearCordAdapter`: SM id to linear byte offset
- `ToRepeatedCordAdapter`: wrap one converted instruction in a prepended `RepeatM.on(...)` bundle while keeping schedule-side modifier calls like `.group()` and `.bar()` targeted at the wrapped TMA instruction
- `ToRopeTableCordAdapter`: SM id to rope-table coordinates
- `ToSplitMCordAdapter`: SM id to `(0, m)` split-M coordinates
- `ToAttnKVStoreCordAdapter`: SM id to attention KV-store coordinates
- `ToAttnVStoreCordAdapter`: GEMV store call shape `(0, m)` to V-cache store coordinates

## Usage Notes

- For copy and rope schedules, wrap every schedule-facing `tma` entry before constructing the schedule.
- Plain cordable objects can be passed directly when their native `.cord(...)` behavior already matches the schedule call shape.
- For GEMV schedules, keep the scheduler logic unchanged, especially `storeC.cord(0, m)`, and adapt only through the wrapper.
- Adapters may now return a small instruction-list proxy instead of a bare instruction; the launcher already flattens lists, and modifier chaining still applies to the designated wrapped instruction.
- The llama and qwen app schedules use small local helpers to wrap common identity/static cases and keep schedule construction readable.

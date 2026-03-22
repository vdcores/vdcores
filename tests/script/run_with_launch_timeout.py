#!/usr/bin/env python3

import argparse
import os
import selectors
import subprocess
import sys
import time
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a command and kill it if it appears to hang after launch. "
            "The timer starts only after the launch marker is observed."
        )
    )
    parser.add_argument(
        "--launch-pattern",
        default="[launch]",
        help="Output substring that marks the start of the launched workload.",
    )
    parser.add_argument(
        "--post-launch-timeout",
        type=float,
        default=60.0,
        help="Maximum total seconds allowed after the launch pattern is observed.",
    )
    parser.add_argument(
        "--post-launch-idle-timeout",
        type=float,
        default=None,
        help="Maximum silent seconds allowed after launch. Defaults to post-launch-timeout.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Optional maximum seconds allowed before the launch pattern is observed.",
    )
    parser.add_argument(
        "--grace-kill-secs",
        type=float,
        default=5.0,
        help="Seconds to wait after terminate() before forcing kill().",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=20,
        help="How many recent output lines to include in timeout diagnostics.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Prefix with -- to separate wrapper args from the command.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command; pass it after --")
    if args.post_launch_idle_timeout is None:
        args.post_launch_idle_timeout = args.post_launch_timeout
    return args


def terminate_process(proc: subprocess.Popen, grace_kill_secs: float):
    if proc.poll() is not None:
        return
    proc.terminate()
    deadline = time.monotonic() + grace_kill_secs
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    if proc.poll() is None:
        proc.kill()


def main():
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    launch_seen = False
    launch_time = None
    last_output_time = time.monotonic()
    recent_lines = deque(maxlen=args.tail_lines)

    try:
        while True:
            now = time.monotonic()
            timeout_reason = None

            if not launch_seen and args.startup_timeout is not None:
                if now - last_output_time > args.startup_timeout:
                    timeout_reason = (
                        f"startup timeout: launch pattern {args.launch_pattern!r} "
                        f"not seen within {args.startup_timeout:.1f}s"
                    )

            if launch_seen and launch_time is not None:
                if now - launch_time > args.post_launch_timeout:
                    timeout_reason = (
                        f"post-launch timeout: process exceeded {args.post_launch_timeout:.1f}s "
                        f"after seeing {args.launch_pattern!r}"
                    )
                elif now - last_output_time > args.post_launch_idle_timeout:
                    timeout_reason = (
                        f"post-launch idle timeout: no output for {args.post_launch_idle_timeout:.1f}s "
                        f"after seeing {args.launch_pattern!r}; this often means a barrier deadlock"
                    )

            if timeout_reason is not None:
                print(f"[timeout] {timeout_reason}", file=sys.stderr)
                if recent_lines:
                    print("[timeout] recent output:", file=sys.stderr)
                    for line in recent_lines:
                        print(line, file=sys.stderr)
                terminate_process(proc, args.grace_kill_secs)
                return 124

            events = selector.select(timeout=0.2)
            if not events:
                if proc.poll() is not None:
                    break
                continue

            for key, _ in events:
                line = key.fileobj.readline()
                if line == "":
                    if proc.poll() is not None:
                        break
                    continue

                line = line.rstrip("\n")
                recent_lines.append(line)
                print(line, flush=True)

                last_output_time = time.monotonic()
                if (not launch_seen) and args.launch_pattern in line:
                    launch_seen = True
                    launch_time = last_output_time
                    print(
                        f"[timeout] launch detected; enforcing post-launch timeout of "
                        f"{args.post_launch_timeout:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )

            if proc.poll() is not None:
                break
    finally:
        selector.unregister(proc.stdout)
        proc.stdout.close()

    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())

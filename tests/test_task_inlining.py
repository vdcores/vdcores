from pathlib import Path


TASK_INCLUDE_DIR = Path(__file__).resolve().parents[1] / "include" / "task"


def test_task_headers_never_disable_inlining():
    offenders = []
    for header in sorted(TASK_INCLUDE_DIR.rglob("*.cuh")):
        for line_number, line in enumerate(
            header.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "noinline" in line.lower():
                offenders.append(
                    f"{header.relative_to(TASK_INCLUDE_DIR)}:{line_number}"
                )

    assert not offenders, (
        "Task device code must remain inline; forbidden noinline annotations: "
        + ", ".join(offenders)
    )

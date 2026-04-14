stage_annotations = {
    "unit": "us",
    "timebase": "relative_from_plot_start",
    "stages": {
        "embedding": {
            "baseline": {"start_us": 0.0, "end_us": 17},
            "vdc": {"start_us": 0.0, "end_us": 3.8},
        },
        "rms-1": {
            "baseline": {"start_us": 17, "end_us": 20},
            "vdc": {"start_us": 0.0, "end_us": 3.8, 'fused': True},
        },
        "qkv": {
            "baseline": {"start_us": 20, "end_us": 41},
            "vdc": {"start_us": 0, "end_us": 17.7},
        },
        "attention": {
            "baseline": {"start_us": 41, "end_us": 54},
            "vdc": {"start_us": 17.8, "end_us": 30.9},
        },
        "o-proj": {
            "baseline": {"start_us": 54, "end_us": 70},
            "vdc": {"start_us": 25, "end_us": 41.0},
        },
        "rms-2": {
            "baseline": {"start_us": 70, "end_us": 73},
            "vdc": {"start_us": 41.0, "end_us": 47.0},
        },
        "gate+up": {
            "baseline": {"start_us": 73, "end_us": 151},
            "vdc": {"start_us": 43.0, "end_us": 120},
        },
        # "gate-1": {
        #     "vdc": {"start_us": 31.0, "end_us": 72.5},
        # },
        # "up-1": {
        #     "vdc": {"start_us": 31.0, "end_us": 72.5},
        # },
        "silu": {
            "baseline": {"start_us": 151, "end_us": 161},
            "vdc": {"start_us": 99, "end_us": 126.0, 'fused': True},
        },
        # "silu-1": {
        #     "vdc": {"start_us": 109, "end_us": 101.0},
        # },
        # "silu-2": {
        #     "vdc": {"start_us": 109, "end_us": 116.0},
        # },
        # "gate-2": {
        #     "vdc": {"start_us": 72.5, "end_us": 109.0},
        # },
        # "up-2": {
        #     "vdc": {"start_us": 72.5, "end_us": 109.0},
        # },
        "down": {
            "baseline": {"start_us": 161, "end_us": 202},
            "vdc": {"start_us": 120, "end_us": 155.0, 'overlap': True},
        }
    },
}

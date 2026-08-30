import argparse
import sys


MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_PREFILL_TOKEN = 51
DEFAULT_DECODE_INPUT_TOKEN = 52
DEBUG_STAGE_ORDER = (
    "qkv",
    "attention",
    "out",
    "post_attn_rms",
    "mlp_prefix",
    "silu_prefix",
    "mlp_tail",
    "down",
    "final_rms",
    "logits",
    "argmax",
    "restore",
    "full",
)


def parse_args():
    raw_argv = sys.argv[1:]
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("-N", "--num-generates", type=int, default=16)
    arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    arg_parser.add_argument("--model", "--model-name", dest="model_name", default=MODEL_NAME)
    arg_parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Logical request batch (1-8); GEMV tiles remain physically N=8",
    )
    arg_parser.add_argument("--correctness", action="store_true")
    arg_parser.add_argument("--debug-num-layers", type=int, default=None)
    arg_parser.add_argument(
        "--debug-stop-after", choices=DEBUG_STAGE_ORDER, default="full"
    )
    parsed_args, remaining_argv = arg_parser.parse_known_args()
    num_generates_explicit = any(arg == "-N" or arg.startswith("--num-generates") for arg in raw_argv)
    if not num_generates_explicit and any(arg in ("-b", "--bench") for arg in remaining_argv):
        parsed_args.num_generates = 1
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    if parsed_args.correctness and (
        parsed_args.debug_num_layers is not None
        or parsed_args.debug_stop_after != "full"
    ):
        raise ValueError("Qwen correctness requires the full model and schedule")
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args

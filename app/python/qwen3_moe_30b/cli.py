import argparse
import sys


MODEL_NAME = "Qwen/Qwen3-30B-A3B"
# MODEL_NAME = "unsloth/Qwen3-30B-A3B-Instruct-2507"
# MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_PREFILL_TOKEN = 51
DEFAULT_DECODE_INPUT_TOKEN = 52


def parse_args():
    raw_argv = sys.argv[1:]
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("-N", "--num-generates", type=int, default=1)
    arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    arg_parser.add_argument("--correctness", action="store_true")
    arg_parser.add_argument("--local-generated-weights", action="store_true")
    arg_parser.add_argument("--synthetic-num-layers", type=int, default=1)
    arg_parser.add_argument("--fixed-top-k", type=int, default=None)
    arg_parser.add_argument("--expert-buffers", type=int, default=None)
    parsed_args, remaining_argv = arg_parser.parse_known_args()
    num_generates_explicit = any(arg == "-N" or arg.startswith("--num-generates") for arg in raw_argv)
    if not num_generates_explicit and any(arg in ("-b", "--bench") for arg in remaining_argv):
        parsed_args.num_generates = 1
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args

from pathlib import Path
from torch.utils.cpp_extension import load

_src_path = Path(__file__).with_name("manual_sum.cu")

_manual_sum_ext = load(
    name="manual_sum_ext",
    sources=[str(_src_path)],
    with_cuda=True,
    verbose=True,
)

manual_reduction = _manual_sum_ext.manual_reduction
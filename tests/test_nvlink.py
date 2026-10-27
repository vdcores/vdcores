"""Run under `na run -n=2 python -m unittest discover -s tests -p test_nvlink.py`."""

import gc
import unittest

import torch

from dae import runtime
from dae.launcher import Launcher
from dae.instructions import Copy, IssueBarrier, RepeatM, TerminateC, TerminateM, TmaLoad1D, TmaStore1D


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two allocated CUDA GPUs")
class PeerLaunchTests(unittest.TestCase):
    def make_copy(self, source, target, device, *, num_gpus=2):
        sms, copies, tile = source.shape
        dae = Launcher(sms, device=device, num_gpus=num_gpus)
        bar = dae.new_bar(sms * copies)
        dae.i(
            Copy(copies, tile),
            lambda sm: RepeatM.on(
                copies,
                [TmaLoad1D(source[sm, 0]), tile],
                [TmaStore1D(target[sm, 0]).bar(bar), tile],
            ),
            TerminateM(), TerminateC(),
        )
        return dae

    def test_explicit_opt_in(self):
        source = torch.zeros((1, 1, 8192), dtype=torch.uint8, device="cuda:1")
        target = torch.empty_like(source, device="cuda:0")
        dae = self.make_copy(source, target, "cuda:0", num_gpus=1)
        with self.assertRaisesRegex(ValueError, "explicit num_gpus"):
            dae.launch()
        for count in (0, -1, True, 1.5, torch.cuda.device_count() + 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Launcher(num_gpus=count)

    def test_peer_pull_and_push_on_noncurrent_device(self):
        for execution_device in ("cuda:0", "cuda:1"):
            for source_device, target_device in (("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")):
                with self.subTest(execution=execution_device, source=source_device):
                    source = torch.randint(0, 256, (4, 7, 8192), dtype=torch.uint8, device=source_device)
                    target = torch.zeros_like(source, device=target_device)
                    dae = self.make_copy(source, target, execution_device)
                    for device in (0, 1):
                        torch.cuda.synchronize(device)
                    current = torch.cuda.current_device()
                    stream = torch.cuda.Stream(device=execution_device)
                    for _ in range(2):
                        dae.reset_barriers()
                        target.zero_()
                        torch.cuda.synchronize(target.device)
                        with torch.cuda.stream(stream):
                            dae.launch()
                        self.assertTrue(torch.equal(source.cpu(), target.cpu()))
                        self.assertEqual(int(dae.bars.view(torch.int32)[0]), 0)
                    self.assertEqual(torch.cuda.current_device(), current)

    def test_instruction_owns_tensor_storage(self):
        source = torch.full((1, 1, 8192), 137, dtype=torch.uint8, device="cuda:1")
        target = torch.zeros_like(source, device="cuda:0")
        dae = self.make_copy(source, target, "cuda:0")
        torch.cuda.synchronize(1)
        del source
        gc.collect()
        dae.launch()
        self.assertTrue(bool((target == 137).all()))

    def test_single_gpu_on_device_one(self):
        source = torch.randint(0, 256, (1, 2, 8192), dtype=torch.uint8, device="cuda:1")
        target = torch.zeros_like(source)
        dae = self.make_copy(source, target, "cuda:1", num_gpus=1)
        dae.launch()
        self.assertTrue(torch.equal(source, target))

    def test_shared_gpu0_barriers(self):
        producer = Launcher(1, device="cuda:0", num_gpus=2)
        consumer = Launcher(1, device="cuda:1", num_gpus=2)
        self.assertIs(producer.bars, consumer.bars)
        self.assertIs(producer.bars_src, consumer.bars_src)
        self.assertEqual(producer.bars.device, torch.device("cuda:0"))
        ready = producer.new_bar(1)
        done = consumer.new_bar(1)
        self.assertNotEqual(ready, done)

        source = torch.randint(256, (8192,), dtype=torch.uint8, device="cuda:0")
        staging = torch.empty_like(source, device="cuda:1")
        result = torch.empty_like(source)
        producer.s(Copy(1, 8192), TmaLoad1D(source),
                   TmaStore1D(staging).bar(ready), IssueBarrier(done))
        consumer.s(Copy(1, 8192), TmaLoad1D(staging).bar(ready),
                   TmaStore1D(result).bar(done))

        def upload(dae):
            dae.build_instructions()
            return (dae.cinsts.to(dae.device).view(-1, 8),
                    dae.minsts.to(dae.device).view(-1, 16),
                    torch.empty((4, 128), dtype=torch.uint8, device=dae.device),
                    dae.profile.view(torch.uint8).view(-1, 8))

        programs = [(dae, upload(dae)) for dae in (consumer, producer)]
        for _ in range(2):
            producer.reset_barriers()
            result.zero_()
            # Start the consumer first: it must wait on the shared GPU-0 counter.
            for dae, (cinsts, minsts, tma, profile) in programs:
                runtime.launch_dae(1, dae.smem_size, cinsts, minsts, tma,
                                   dae.bars, profile,
                                   torch.cuda.current_stream(dae.device).cuda_stream, False)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            self.assertTrue(torch.equal(result, source))
            self.assertEqual(int(producer.bars.view(torch.int32)[ready]), 0)
            self.assertEqual(int(producer.bars.view(torch.int32)[done]), 0)
            self.assertEqual(int(producer.bars_src.view(torch.int32)[ready]), 1)
            self.assertEqual(int(producer.bars_src.view(torch.int32)[done]), 1)


if __name__ == "__main__":
    unittest.main()

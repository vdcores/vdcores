// Standalone ASan/UBSan/TSan harness, independent of Python's runtime.
#include "../../src/embedded_loop.cpp"
#include <cassert>
#include <iostream>

int main() {
  constexpr uint64_t n = 257, frames = 500, lanes = 8, workers = 4;
  std::vector<std::vector<double>> buffers(lanes, std::vector<double>(n));
  std::vector<double*> pointers;
  for (auto& buffer : buffers) pointers.push_back(buffer.data());
  void* handle = el_create(pointers.data(), n, frames, lanes, workers);
  assert(handle);
  uint64_t completed = 0;
  while (completed < frames * lanes) {
    std::vector<Request> requests;
    Request request{};
    assert(el_take(handle, &request, 1000) == 1);
    requests.push_back(request);
    while (el_take(handle, &request, 0) == 1) requests.push_back(request);
    // Independent lanes may complete in the opposite order to dispatch.
    for (auto it = requests.rbegin(); it != requests.rend(); ++it) {
      const auto value = el_reduce(pointers[it->lane], n);
      if (it->frame % 2 == 0)
        assert(el_complete(handle, it->lane, it->frame, value) == 0);
      else
        assert(el_dispatch_cpu(handle, it->lane, it->frame) == 0);
      assert(el_complete(handle, it->lane, it->frame, value) == -1);
      ++completed;
    }
  }
  assert(el_wait(handle, 1000) == 1);
  std::vector<Trace> traces(frames * lanes);
  assert(el_traces(handle, traces.data()) == 0);
  for (const auto& t : traces) {
    assert(t.ready_ns <= t.prep_start_ns && t.prep_start_ns <= t.prep_end_ns);
    assert(t.prep_end_ns <= t.taken_ns && t.taken_ns <= t.external_end_ns);
    assert(t.external_end_ns <= t.control_start_ns && t.control_start_ns <= t.end_ns);
    assert(std::isfinite(t.action));
  }
  el_destroy(handle);
  // Cancel while workers may still be writing, then join before buffer release.
  for (int i = 0; i < 100; ++i) {
    handle = el_create(pointers.data(), n, frames, lanes, workers);
    assert(handle);
    el_cancel(handle);
    el_destroy(handle);
  }
  std::cout << "PASS: 4000 feedback frames, reordered completions, 100 cancellations\n";
}

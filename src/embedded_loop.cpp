// Fixed-graph CPU Virtual Core prototype. No CUDA/Torch dependency.
// Buffer ownership: native PREP -> external backend -> native CONTROL -> PREP.
// Every transition and completion publication is protected by mutex_. Native
// writes before unlock happen-before the backend's take; backend completion
// happens-before the controller reads its result. A lane never has two frames
// in flight, and cancellation is terminal.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count();
}
thread_local std::string last_error;
struct Request { uint64_t lane, frame; };
struct Trace {
  uint64_t lane, frame;
  double previous_action, feature, action;
  int64_t ready_ns, prep_start_ns, prep_end_ns, taken_ns;
  int64_t external_end_ns, control_start_ns, end_ns;
  uint64_t prep_worker, control_worker;
};
enum class State { prep_queued, prep_running, external_ready, external_running,
                   feature_queued, feature_running, control_queued, control_running, done };
struct Lane {
  double* input;
  uint64_t frame = 0;
  double action = 0;
  State state = State::prep_queued;
};

void preprocess(double* output, uint64_t count, uint64_t lane,
                uint64_t frame, double previous) {
  // Deterministic sensor surrogate; previous control output changes next input.
  for (uint64_t i = 0; i < count; ++i)
    output[i] = (static_cast<int>(i % 17) - 8) / 16.0 +
                previous * 0.25 + (frame % 7) * 0.001 + lane * 0.0001;
}
double reduce(const double* input, uint64_t count) {
  double sum = 0;
  for (uint64_t i = 0; i < count; ++i) sum += input[i] * input[i];
  return sum / static_cast<double>(count);
}

class Loop {
 public:
  Loop(double** inputs, uint64_t count, uint64_t frames, uint64_t lanes,
       uint64_t workers) : count_(count), frames_(frames), traces_(frames * lanes),
                           worker_tasks_(workers, 0) {
    if (!inputs || !count || !frames || !lanes || !workers)
      throw std::invalid_argument("sizes and worker count must be positive");
    for (uint64_t i = 0; i < lanes; ++i) {
      if (!inputs[i]) throw std::invalid_argument("null input buffer");
      lanes_.push_back(Lane{inputs[i]});
      trace(i).ready_ns = now_ns();
      cpu_ready_.push_back(i);
      ++cpu_preps_;
    }
    try {
      for (uint64_t i = 0; i < workers; ++i)
        workers_.emplace_back([this, i] { work(i); });
    } catch (...) {
      cancel();
      for (auto& worker : workers_) worker.join();
      throw;
    }
  }
  ~Loop() {
    cancel();
    for (auto& worker : workers_) worker.join();
  }
  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
  }
  int take(Request* request, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    // A zero-time poll must not enter pthread_cond_timedwait: even an expired
    // wait can deschedule the service thread and delay short CUDA completions.
    if (timeout_ms > 0)
      cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return stopped_ || !external_ready_.empty() || done_ == lanes_.size();
      });
    check_live();
    if (external_ready_.empty()) return 0;
    const auto lane = external_ready_.front();
    external_ready_.pop_front();
    lanes_[lane].state = State::external_running;
    trace(lane).taken_ns = now_ns();
    *request = Request{lane, lanes_[lane].frame};
    return 1;
  }
  void complete(uint64_t lane, uint64_t frame, double feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_live();
    if (lane >= lanes_.size()) throw std::invalid_argument("unknown lane");
    auto& slot = lanes_[lane];
    if (slot.frame != frame || slot.state != State::external_running)
      throw std::invalid_argument("stale, duplicate, or unissued completion token");
    if (!std::isfinite(feature)) throw std::invalid_argument("nonfinite feature");
    auto& t = trace(lane);
    t.feature = feature;
    t.external_end_ns = now_ns();
    slot.state = State::control_queued;
    ++cpu_controls_;
    cpu_ready_.push_back(lane);
    cv_.notify_all();
  }
  void dispatch_cpu(uint64_t lane, uint64_t frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_live();
    if (lane >= lanes_.size()) throw std::invalid_argument("unknown lane");
    auto& slot = lanes_[lane];
    if (slot.frame != frame || slot.state != State::external_running)
      throw std::invalid_argument("stale, duplicate, or unissued CPU dispatch token");
    slot.state = State::feature_queued;
    ++cpu_features_;
    cpu_ready_.push_back(lane);
    cv_.notify_all();
  }
  int cpu_load() {
    std::lock_guard<std::mutex> lock(mutex_);
    check_live();
    return static_cast<int>(cpu_features_);
  }
  void cpu_backlog(uint64_t* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_live();
    output[0] = cpu_preps_;
    output[1] = cpu_features_;
    output[2] = cpu_controls_;
  }
  int wait(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout_ms > 0)
      cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return stopped_ || done_ == lanes_.size();
      });
    check_live();
    return done_ == lanes_.size();
  }
  void copy_traces(Trace* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_live();
    if (done_ != lanes_.size()) throw std::runtime_error("loop is not complete");
    std::copy(traces_.begin(), traces_.end(), output);
  }
  void counts(uint64_t* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::copy(worker_tasks_.begin(), worker_tasks_.end(), output);
  }

 private:
  Trace& trace(uint64_t lane) { return traces_[lane * frames_ + lanes_[lane].frame]; }
  void check_live() {
    if (stopped_) throw std::runtime_error(error_.empty() ? "loop cancelled" : error_);
  }
  void work(uint64_t worker) noexcept {
    try {
      for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopped_ || !cpu_ready_.empty(); });
        if (stopped_) return;
        const auto lane = cpu_ready_.front();
        cpu_ready_.pop_front();
        auto& slot = lanes_[lane];
        auto& t = trace(lane);
        ++worker_tasks_[worker];
        if (slot.state == State::prep_queued) {
          slot.state = State::prep_running;
          t.lane = lane;
          t.frame = slot.frame;
          t.previous_action = slot.action;
          t.prep_worker = worker;
          t.prep_start_ns = now_ns();
          lock.unlock();
          preprocess(slot.input, count_, lane, slot.frame, slot.action);
          lock.lock();
          if (stopped_) return;
          t.prep_end_ns = now_ns();
          --cpu_preps_;
          slot.state = State::external_ready;
          external_ready_.push_back(lane);
        } else if (slot.state == State::feature_queued) {
          slot.state = State::feature_running;
          lock.unlock();
          const auto feature = reduce(slot.input, count_);
          lock.lock();
          --cpu_features_;
          if (stopped_) return;
          t.feature = feature;
          t.external_end_ns = now_ns();
          slot.state = State::control_queued;
          ++cpu_controls_;
          cpu_ready_.push_back(lane);
        } else if (slot.state == State::control_queued) {
          slot.state = State::control_running;
          t.control_worker = worker;
          t.control_start_ns = now_ns();
          slot.action = std::clamp(0.9 * slot.action - 0.5 * t.feature, -1.0, 1.0);
          t.action = slot.action;
          t.end_ns = now_ns();
          --cpu_controls_;
          if (slot.frame + 1 == frames_) {
            slot.state = State::done;
            ++done_;
          } else {
            ++slot.frame;
            slot.state = State::prep_queued;
            ++cpu_preps_;
            trace(lane).ready_ns = t.end_ns;
            cpu_ready_.push_back(lane);
          }
        } else {
          throw std::runtime_error("invalid native CPU stage transition");
        }
        cv_.notify_all();
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = e.what();
      stopped_ = true;
      cv_.notify_all();
    }
  }
  uint64_t count_, frames_, done_ = 0, cpu_features_ = 0;
  uint64_t cpu_preps_ = 0, cpu_controls_ = 0;
  std::vector<Trace> traces_;
  std::vector<Lane> lanes_;
  std::vector<uint64_t> worker_tasks_;
  std::deque<uint64_t> cpu_ready_, external_ready_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_ = false;
  std::string error_;
};
template <typename F> int checked(F&& f) noexcept {
  try { return f(); }
  catch (const std::exception& e) { last_error = e.what(); return -1; }
  catch (...) { last_error = "unknown native runtime failure"; return -1; }
}
}  // namespace

extern "C" {
int el_abi_version() { return 3; }
const char* el_error() { return last_error.c_str(); }
void* el_create(double** inputs, uint64_t count, uint64_t frames,
                uint64_t lanes, uint64_t workers) noexcept {
  try {
    if (!count || count > 16777216 || !frames || frames > 1000000 ||
        !lanes || lanes > 256 || !workers || workers > 256 ||
        frames * lanes > 1000000 || count * lanes > 33554432)
      throw std::invalid_argument("native loop sizes exceed prototype limits");
    return new Loop(inputs, count, frames, lanes, workers);
  }
  catch (const std::exception& e) { last_error = e.what(); return nullptr; }
  catch (...) { last_error = "native loop creation failed"; return nullptr; }
}
void el_destroy(void* handle) { delete static_cast<Loop*>(handle); }
void el_cancel(void* handle) { static_cast<Loop*>(handle)->cancel(); }
int el_take(void* h, Request* r, int timeout_ms) {
  return checked([&] { return static_cast<Loop*>(h)->take(r, timeout_ms); });
}
int el_complete(void* h, uint64_t lane, uint64_t frame, double value) {
  return checked([&] { static_cast<Loop*>(h)->complete(lane, frame, value); return 0; });
}
int el_dispatch_cpu(void* h, uint64_t lane, uint64_t frame) {
  return checked([&] { static_cast<Loop*>(h)->dispatch_cpu(lane, frame); return 0; });
}
int el_cpu_load(void* h) {
  return checked([&] { return static_cast<Loop*>(h)->cpu_load(); });
}
int el_cpu_backlog(void* h, uint64_t* output) {
  return checked([&] { static_cast<Loop*>(h)->cpu_backlog(output); return 0; });
}
int el_wait(void* h, int timeout_ms) {
  return checked([&] { return static_cast<Loop*>(h)->wait(timeout_ms); });
}
int el_traces(void* h, Trace* output) {
  return checked([&] { static_cast<Loop*>(h)->copy_traces(output); return 0; });
}
int el_worker_counts(void* h, uint64_t* output) {
  return checked([&] { static_cast<Loop*>(h)->counts(output); return 0; });
}
double el_reduce(const double* input, uint64_t count) { return reduce(input, count); }
}

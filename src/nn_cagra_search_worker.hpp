#ifndef FAISSR_NN_CAGRA_SEARCH_WORKER_HPP
#define FAISSR_NN_CAGRA_SEARCH_WORKER_HPP

#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "nn_cagra_merge.hpp"

namespace faissr {

class CagraSearchWorker {
 public:
  CagraSearchWorker() : thread_([this] { run(); }) {}

  ~CagraSearchWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    thread_.join();
  }

  template <typename Task>
  std::future<CagraBatch> submit(Task&& work) {
    std::packaged_task<CagraBatch()> task(std::forward<Task>(work));
    auto result = task.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (pending_.valid()) {
        throw std::logic_error("A CAGRA shard request is already queued");
      }
      pending_ = std::move(task);
    }
    ready_.notify_one();
    return result;
  }

 private:
  void run() {
    for (;;) {
      std::packaged_task<CagraBatch()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || pending_.valid(); });
        if (!pending_.valid()) return;
        task = std::move(pending_);
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::packaged_task<CagraBatch()> pending_;
  bool stopping_ = false;
  std::thread thread_;
};

// The writer must touch only the supplied output slot, with no R API calls.
template <typename Write>
int merge_cagra_rows(CagraSearchWorker* worker,
    const std::vector<CagraBatch>& batches,
    const std::vector<CagraShard>& shards, int first, int rows, int k,
    int stride, int output_first, Write write) {
  const auto merge = [&](int begin, int count) {
    merge_cagra_candidates(batches, shards, first + begin, count, k,
      stride, output_first + begin, write, begin);
  };
  if (!worker || rows < 128) {
    merge(0, rows);
    return 1;
  }
  const int half = rows / 2;
  auto task = worker->submit([&] {
    merge(0, half);
    return CagraBatch{};
  });
  try {
    merge(half, rows - half);
  } catch (...) {
    task.wait();
    throw;
  }
  task.get();
  return 2;
}

}  // namespace faissr

#endif

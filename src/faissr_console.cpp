#ifdef FAISSR_FAISS_CPU_ONLY
#include "faissr_console.hpp"

#include <R_ext/Print.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {
struct Message {
  bool error;
  std::string text;
};
std::mutex console_mutex;
std::vector<Message> messages;
std::size_t message_bytes = 0;
constexpr std::size_t message_limit = 1024 * 1024;
thread_local unsigned int console_depth = 0;

bool is_console(FILE* stream) {
  return stream == faissr_console_stdout() ||
    stream == faissr_console_stderr();
}

int buffer_message(FILE* stream, const char* format, va_list arguments) {
  if (!is_console(stream)) return std::vfprintf(stream, format, arguments);
  va_list count_arguments;
  va_copy(count_arguments, arguments);
  const int count = std::vsnprintf(nullptr, 0, format, count_arguments);
  va_end(count_arguments);
  if (count <= 0) return count;
  const std::size_t size = static_cast<std::size_t>(count);
  std::lock_guard<std::mutex> lock(console_mutex);
  if (message_bytes >= message_limit) return count;
  const std::size_t available = message_limit - message_bytes;
  const std::size_t kept = size < available ? size : available;
  std::vector<char> buffer(kept + 1);
  std::vsnprintf(buffer.data(), buffer.size(), format, arguments);
  messages.push_back({stream == faissr_console_stderr(),
    std::string(buffer.data(), kept)});
  message_bytes += kept;
  return count;
}
}

// These sentinels are never passed to the C stdio implementation.
FILE* faissr_console_stdout() {
  return reinterpret_cast<FILE*>(static_cast<std::intptr_t>(-1));
}
FILE* faissr_console_stderr() {
  return reinterpret_cast<FILE*>(static_cast<std::intptr_t>(-2));
}

int faissr_console_printf(const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);
  const int count = buffer_message(faissr_console_stdout(), format, arguments);
  va_end(arguments);
  return count;
}

int faissr_console_fprintf(FILE* stream, const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);
  const int count = buffer_message(stream, format, arguments);
  va_end(arguments);
  return count;
}

int faissr_console_puts(const char* text) {
  return faissr_console_printf("%s\n", text);
}
int faissr_console_putchar(int value) {
  faissr_console_printf("%c", value);
  return value;
}
int faissr_console_fputs(const char* text, FILE* stream) {
  return faissr_console_fprintf(stream, "%s", text);
}
int faissr_console_fputc(int value, FILE* stream) {
  faissr_console_fprintf(stream, "%c", value);
  return value;
}
int faissr_console_fflush(FILE* stream) {
  return is_console(stream) ? 0 : std::fflush(stream);
}

faissr::ConsoleScope::ConsoleScope() { ++console_depth; }

faissr::ConsoleScope::~ConsoleScope() {
  if (--console_depth != 0) return;
  std::vector<Message> pending;
  bool truncated;
  {
    std::lock_guard<std::mutex> lock(console_mutex);
    pending.swap(messages);
    truncated = message_bytes >= message_limit;
    message_bytes = 0;
  }
  // FAISS workers only buffer text; only the calling R thread uses R's API.
  for (const auto& message : pending) {
    if (message.error) REprintf("%s", message.text.c_str());
    else Rprintf("%s", message.text.c_str());
  }
  if (truncated) REprintf("FAISS diagnostic output truncated after 1 MiB.\n");
}
#endif

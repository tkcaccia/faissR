#ifndef FAISSR_CONSOLE_HPP
#define FAISSR_CONSOLE_HPP

#include <cstdarg>
#include <cstdio>

namespace faissr {
class ConsoleScope {
 public:
  ConsoleScope();
  ~ConsoleScope();
  ConsoleScope(const ConsoleScope&) = delete;
  ConsoleScope& operator=(const ConsoleScope&) = delete;
};
}

int faissr_console_printf(const char* format, ...);
int faissr_console_fprintf(FILE* stream, const char* format, ...);
int faissr_console_puts(const char* text);
int faissr_console_putchar(int value);
int faissr_console_fputs(const char* text, FILE* stream);
int faissr_console_fputc(int value, FILE* stream);
int faissr_console_fflush(FILE* stream);
FILE* faissr_console_stdout();
FILE* faissr_console_stderr();

#ifdef FAISSR_CONSOLE_REMAP
#define printf(...) faissr_console_printf(__VA_ARGS__)
#define fprintf(...) faissr_console_fprintf(__VA_ARGS__)
#define puts(...) faissr_console_puts(__VA_ARGS__)
#define putchar(...) faissr_console_putchar(__VA_ARGS__)
#define fputs(...) faissr_console_fputs(__VA_ARGS__)
#define fputc(...) faissr_console_fputc(__VA_ARGS__)
#define fflush(...) faissr_console_fflush(__VA_ARGS__)
#undef stdout
#undef stderr
#define stdout faissr_console_stdout()
#define stderr faissr_console_stderr()
#endif

#endif

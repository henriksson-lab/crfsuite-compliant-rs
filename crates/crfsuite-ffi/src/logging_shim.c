#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* Rust callback type: receives formatted string + length, returns 0 to continue, non-zero to cancel */
typedef int (*rust_message_callback)(void *user, const char *message, int len);

/* Logging callback that prints to stdout (matches original C CLI behavior) */
int crfsuite_ffi_logging_stdout(void *user, const char *format, va_list args) {
    (void)user;
    vfprintf(stdout, format, args);
    fflush(stdout);
    return 0;
}

/* Logging callback that formats and forwards to a Rust callback.
   The user pointer must point to a struct { rust_message_callback cb; void *rust_user; } */
typedef struct {
    rust_message_callback cb;
    void *rust_user;
} rust_callback_data;

int crfsuite_ffi_logging_to_rust(void *user, const char *format, va_list args) {
    rust_callback_data *data = (rust_callback_data *)user;
    char buf[4096];
    int n = vsnprintf(buf, sizeof(buf), format, args);
    if (n > 0 && data && data->cb) {
        if ((size_t)n >= sizeof(buf)) n = sizeof(buf) - 1;
        return data->cb(data->rust_user, buf, n);
    }
    return 0;
}

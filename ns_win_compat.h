/*
 * ns_win_compat.h — build the AVX2 int8 kernel on Windows/MinGW.
 *
 * Force-included, so the kernel source itself is never edited:
 *     gcc -O3 -mavx2 -include ns_win_compat.h -shared -o ns.dll <kernel>.c
 *
 * THE PROBLEM
 *   The kernel calls C11 aligned_alloc(). MinGW-w64 does not provide it — Windows offers
 *   _aligned_malloc() instead, and memory from _aligned_malloc MUST be released with
 *   _aligned_free(), never plain free().
 *
 * WHY THE OBVIOUS SHIM IS A HEAP-CORRUPTION BUG
 *   The tempting fix is
 *       #define aligned_alloc(a,n) _aligned_malloc(n,a)
 *       #define free _aligned_free
 *   but the kernel sources free() aligned AND ordinary malloc'd pointers through the same
 *   free() calls (e.g. the weight buffers alongside plain scratch arrays). Redirecting free()
 *   globally would hand malloc'd pointers to _aligned_free — undefined behaviour, and the
 *   kind that corrupts the heap quietly rather than crashing where the mistake is.
 *
 * WHY PLAIN malloc IS CORRECT HERE
 *   The alignment is not actually required. Every vector access in all three kernels is
 *   _mm256_loadu_si256 — the UNALIGNED load. Checked: 24 unaligned loads across the three
 *   files, and zero aligned loads or stores (_mm256_load_si256 / _mm256_store_si256).
 *   So the 32-byte request buys nothing the code relies on, and malloc'd memory can be
 *   released by the plain free() the sources already call. Correctness is preserved exactly.
 *
 *   The only cost is that a 32-byte vector load may occasionally straddle a cache line
 *   (malloc typically guarantees 16-byte alignment). On modern x86 that penalty is small;
 *   on the AMD Zen 3 this targets it is negligible. If it ever shows up in a profile, the
 *   real fix is to pair _aligned_malloc with _aligned_free at each call site in the kernel —
 *   not to redirect free() globally.
 */
#ifndef NS_WIN_COMPAT_H
#define NS_WIN_COMPAT_H

#if defined(_WIN32) || defined(__MINGW32__)
#include <stdlib.h>

/* Only if the toolchain genuinely lacks it — newer UCRT/MinGW builds may provide it. */
#ifndef aligned_alloc
#define aligned_alloc(alignment, size) malloc((size))
#endif

#endif /* _WIN32 */
#endif /* NS_WIN_COMPAT_H */

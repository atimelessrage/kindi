// SPDX-License-Identifier: AGPL-3.0-only
//
// The SYCL vector-add smoke for the Intel Arc A770 (REQ-0020 M2 evidence).
//
// It is deliberately not a Kindi kernel: the M2 gate asks whether the SYCL runtime and
// SYCL device selection reach the card, and one exact vector-add answers only that. The
// Python driver (`kindi/intel_sycl.py`) builds this file with the oneAPI DPC++ compiler,
// runs it under `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and validates the JSON it prints.
//
// The child refuses to pass on any backend but Level Zero: a pass that silently used the
// OpenCL backend or the host device would be a false M2 signal.
//
// Build (after a Developer Prompt, so MSVC's headers and linker are on PATH):
//   icpx -fsycl -O2 -o intel_sycl_vecadd.exe tools/intel_sycl_vecadd.cpp
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./intel_sycl_vecadd.exe --compiler-version "..."

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr size_t kBig = 1048576;        // 1 Mi elements, five timed repeats
constexpr size_t kTiny = 1;             // the degenerate launch
constexpr size_t kOdd = 1000003;        // not a multiple of the local size: the guard runs
constexpr size_t kLocal = 64;

struct SizeResult {
  size_t n = 0;
  int repeats = 0;
  size_t mismatches = 0;
  double max_abs_error = 0.0;
  double elapsed_ms = 0.0;
  std::string error;
};

// Integer-valued inputs, so every sum is exact in float32 and the comparison can be
// bit-for-bit rather than tolerance-based.
inline float input_a(size_t i) { return static_cast<float>((31 * i) % 1000); }
inline float input_b(size_t i) { return static_cast<float>((17 * i) % 1000); }

std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') { out += "\\n"; }
    else { out += c; }
  }
  return out;
}

void print_failure(const std::string &stage, const std::string &error,
                   const std::string &compiler, const std::string &selector) {
  std::printf("{\n");
  std::printf("  \"probe\": \"intel-sycl-smoke\",\n");
  std::printf("  \"schema\": 1,\n");
  std::printf("  \"support_claim\": false,\n");
  std::printf("  \"scope\": \"SYCL runtime and device selection on this Windows host; not a model, engine or Kindi kernel\",\n");
  std::printf("  \"toolchain\": {\"compiler\": \"%s\", \"device_selector\": \"%s\"},\n",
              json_escape(compiler).c_str(), json_escape(selector).c_str());
  std::printf("  \"device\": null,\n");
  std::printf("  \"vector_add\": {\"sizes\": []},\n");
  std::printf("  \"result\": \"fail\",\n");
  std::printf("  \"stage\": \"%s\",\n", json_escape(stage).c_str());
  std::printf("  \"error\": \"%s\"\n", json_escape(error).c_str());
  std::printf("}\n");
}

bool run_one(sycl::queue &q, size_t n, int repeats, SizeResult &out) {
  out.n = n;
  out.repeats = repeats;
  float *a = sycl::malloc_shared<float>(n, q);
  float *b = sycl::malloc_shared<float>(n, q);
  float *c = sycl::malloc_shared<float>(n, q);
  if (a == nullptr || b == nullptr || c == nullptr) {
    out.error = "malloc_shared returned null";
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return false;
  }
  for (size_t i = 0; i < n; ++i) { a[i] = input_a(i); b[i] = input_b(i); c[i] = -12345.0f; }

  const size_t groups = (n + kLocal - 1) / kLocal;
  const auto start = std::chrono::steady_clock::now();
  std::string launch_error;
  for (int r = 0; r < repeats; ++r) {
    try {
      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * kLocal), sycl::range<1>(kLocal)),
                       [=](sycl::nd_item<1> it) {
                         const size_t i = it.get_global_id(0);
                         if (i < n) { c[i] = a[i] + b[i]; }
                       });
      });
      q.wait_and_throw();
    } catch (const sycl::exception &e) {
      launch_error = e.what();
      break;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  out.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  if (!launch_error.empty()) {
    out.error = launch_error;
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    const float want = input_a(i) + input_b(i);
    if (c[i] != want) {
      ++out.mismatches;
      const double err = std::fabs(static_cast<double>(c[i]) - static_cast<double>(want));
      if (err > out.max_abs_error) { out.max_abs_error = err; }
    }
  }
  sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  // Both values come from the driver rather than the environment, so the report records what
  // the driver actually used instead of re-reading a variable it may not have set.
  std::string compiler = "unknown";
  std::string selector;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--compiler-version") == 0 && i + 1 < argc) { compiler = argv[++i]; }
    else if (std::strcmp(argv[i], "--selector") == 0 && i + 1 < argc) { selector = argv[++i]; }
  }

  // Enumerate every GPU the runtime offers and keep only the Level Zero one. Listing them in
  // the failure message is the useful part: it says whether the L0 adapter loaded at all.
  sycl::device dev;
  bool found = false;
  std::string seen;
  try {
    for (const sycl::device &candidate : sycl::device::get_devices(sycl::info::device_type::gpu)) {
      if (!seen.empty()) { seen += "; "; }
      seen += candidate.get_info<sycl::info::device::name>() + " (" +
              (candidate.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" :
               candidate.get_backend() == sycl::backend::opencl ? "opencl" : "other") + ")";
      if (candidate.get_backend() == sycl::backend::ext_oneapi_level_zero && candidate.is_gpu()) {
        dev = candidate;
        found = true;
        break;
      }
    }
  } catch (const sycl::exception &e) {
    print_failure("device", std::string("device enumeration failed: ") + e.what(), compiler, selector);
    return 1;
  }
  if (!found) {
    // No Level Zero GPU: refuse rather than pass on the OpenCL backend or the host device.
    print_failure("device", "no level_zero:gpu device (runtime offered: " +
                               (seen.empty() ? std::string("none") : seen) + ")",
                  compiler, selector);
    return 1;
  }

  const std::string backend = "ext_oneapi_level_zero";
  const std::string name = dev.get_info<sycl::info::device::name>();
  const std::string vendor = dev.get_info<sycl::info::device::vendor>();
  const std::string driver = dev.get_info<sycl::info::device::driver_version>();
  const size_t global_mib = dev.get_info<sycl::info::device::global_mem_size>() / (1024u * 1024u);
  const size_t local_max = dev.get_info<sycl::info::device::local_mem_size>();

  sycl::queue q(dev, sycl::property::queue::in_order());
  std::vector<SizeResult> results(3);
  const bool big_ok = run_one(q, kBig, 5, results[0]);
  const bool tiny_ok = big_ok && run_one(q, kTiny, 5, results[1]);
  const bool odd_ok = tiny_ok && run_one(q, kOdd, 5, results[2]);
  const bool exact = big_ok && tiny_ok && odd_ok &&
                     results[0].mismatches == 0 && results[1].mismatches == 0 &&
                     results[2].mismatches == 0;
  const bool a770 = name.find("A770") != std::string::npos;

  std::printf("{\n");
  std::printf("  \"probe\": \"intel-sycl-smoke\",\n");
  std::printf("  \"schema\": 1,\n");
  std::printf("  \"support_claim\": false,\n");
  std::printf("  \"scope\": \"SYCL runtime and device selection on this Windows host; not a model, engine or Kindi kernel\",\n");
  std::printf("  \"toolchain\": {\"compiler\": \"%s\", \"device_selector\": \"%s\"},\n",
              json_escape(compiler).c_str(), json_escape(selector).c_str());
  std::printf("  \"device\": {\"backend\": \"%s\", \"name\": \"%s\", \"vendor\": \"%s\", "
              "\"driver\": \"%s\", \"global_mem_mib\": %zu, \"local_mem_kb\": %zu, \"is_gpu\": true},\n",
              backend.c_str(), json_escape(name).c_str(), json_escape(vendor).c_str(),
              json_escape(driver).c_str(), global_mib, local_max / 1024u);
  std::printf("  \"vector_add\": {\"local_size\": %zu, \"sizes\": [\n", kLocal);
  for (size_t i = 0; i < results.size(); ++i) {
    const SizeResult &r = results[i];
    std::printf("    {\"n\": %zu, \"repeats\": %d, \"mismatches\": %zu, \"max_abs_error\": %.1f, "
                "\"elapsed_ms\": %.2f, \"error\": \"%s\"}%s\n",
                r.n, r.repeats, r.mismatches, r.max_abs_error, r.elapsed_ms,
                json_escape(r.error).c_str(), i + 1 == results.size() ? "" : ",");
  }
  std::printf("  ]},\n");
  const bool pass = exact && a770;
  std::printf("  \"result\": \"%s\",\n", pass ? "pass" : "fail");
  std::printf("  \"stage\": \"%s\",\n", pass ? "done" : (a770 ? "exactness" : "device-name"));
  std::printf("  \"error\": \"%s\"\n",
              pass ? "" : (a770 ? "the vector-add is not exact" : "the Level Zero device is not the A770"));
  std::printf("}\n");
  return pass ? 0 : 1;
}

# Lessons learned

Append-only notes from verified work. A lesson should name the surprising observation,
the evidence, and the guard or practice that prevents a repeat. Do not turn an untested
hypothesis into a lesson.

## L-0001: A narrow kernel guard is better than a broad performance claim

**Observation:** The 4B tensor-core plan with `KINDI_FD=1` was 2/4 against llama.cpp, while
the same plan with the split-decode fallback was 3 exact + 1 accepted tie.

**Evidence:** `kindi verify` on the RTX 4060 Ti; the forced-FD failure first appeared at
prompt 2 token 6 and prompt 3 token 37. The fallback is covered by
`tests/test_plan_engine.py::test_split_flash_decode_is_disabled_for_the_padded_4b`.

**Practice:** When a geometry-specific optimization diverges, guard the geometry explicitly
and record the measured limitation. Do not generalize the kernel change or describe the
path as fully supported until a new exactness run says so.

## L-0002: Performance caches need the same key set as the live layout

**Observation:** A streamed `g5` cache written before the tied-head alias had
`output.weight` but no `token_embd.weight`. The old load path accepted it and the plan
build failed with `KeyError: 'token_embd.weight'`.

**Evidence:** The cache JSON contained 201 tensors including `output.weight`; the current
slice set required `token_embd.weight`. The repair is covered by
`tests/test_plan_engine.py::test_streamed_g5_cache_is_rejected_when_a_tied_head_key_is_missing`.

**Practice:** A cache is valid only if its key set covers the current slices. Rebuild on a
missing key; never treat cache presence as proof of compatibility.

## L-0003: CPU thread defaults are topology decisions, not "use all cores"

**Observation:** The i7-8086K 2B engine collapsed from about 39 tokens/s at 6 threads to
about 1.3 at 12. The cause was libgomp spinning both hyperthread siblings at the per-row
barriers.

**Evidence:** `docs/CPU-SUPPORT.md` and the thread sweep; the default is one thread per
physical core, with `tests/test_cpu_threads.py` pinning the topology parser.

**Practice:** Do not let OpenMP default to every logical CPU. Measure a thread sweep before
changing the default, and keep the result hardware-specific.

## L-0004: Generated references move when line numbers move

**Observation:** Adding a guard in `kindi/spec.py` shifted source line references in
`docs/CONFIGURATION-REFERENCE.md`; the full suite failed only after the code edit.

**Evidence:** `python -m kindi.config_reference --check` failed with stale source lines
until the file was regenerated.

**Practice:** After any change to the files read by a generator, run the generator, not
just the tests. The generated file's `--check` is part of the handoff gate.

## L-0005: Do not let a documentation edit rewrite line endings

**Observation:** A documentation tool edit produced CRCRLF line endings across a Markdown
file. `git diff --check` then flagged every line as trailing whitespace even though the
content was correct.

**Evidence:** `git diff --check` reported the whole file; byte inspection showed
`\r\r\n` where `HEAD` had `\n`.

**Practice:** After editing an existing Markdown file on Windows, run `git diff --check`
and inspect the byte-level line endings if the diff is unexpectedly large. New files should
use LF; do not normalize unrelated lines.

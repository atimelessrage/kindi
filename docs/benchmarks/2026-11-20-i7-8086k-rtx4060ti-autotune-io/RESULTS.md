# Memory-tier rates on the i7-8086K + RTX 4060 Ti

Measured 2026-11-20 with `python -m kindi io --mib 64 --json io.json` on the
i7-8086K (6 physical cores) + GeForce RTX 4060 Ti 8 GB, driver 617.42, native
Windows, Python 3.13.6.

| probe | rate | method |
|---|---:|---|
| RAM copy | **13,266.4 MiB/s** | `memcpy` into a second buffer, 1 thread |
| disk read | **2,547.5 MiB/s** | `FILE_FLAG_NO_BUFFERING`, 64 MiB temporary file |
| PCIe H2D | **6.0 GiB/s** | `cudaMemcpy`, pinned host, 5 timed runs |
| PCIe D2H | **6.1 GiB/s** | `cudaMemcpy`, pinned host, 5 timed runs |

Raw report: [io.json](io.json). The full command output is [run.log](run.log); its
interpreter path is shortened to `<python>`.

This is one run on this host and storage; the rates are inputs to slot sizing, not
targets or claims about another machine. The disk rate is unbuffered as reported by
the probe (`"unbuffered": true`). No model was loaded and no throughput was measured.

"""Read a GGUF header without downloading the weights, and print what it says.

A 27B GGUF is 6 GB; its header (metadata plus the tensor table) is a few hundred kilobytes. An HTTP
range request for the first 12 MB is enough to answer the questions that decide whether an engine can
load a file: architecture, geometry, tensor types, tensor names, vocabulary, context and the chat
template. This is the script that produced `header.json` here from
`https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/resolve/main/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`
with `Range: bytes=0-12582911` (HTTP 206).

    python parse_header.py <gguf-or-header-prefix> > header.json

The file may be truncated: only metadata and the tensor table are read, never tensor data.
"""
import collections
import json
import struct
import sys

S = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
TYPE_NAMES = {0: "f32", 143: "PTQ1_0", 142: "PQ2_0", 42: "Q2_0", 30: "bf16", 1: "f16"}
KEEP = ("general.architecture", "general.name", "general.size_label", "general.file_type",
        "general.quantization_version", "general.sampling.temp", "general.sampling.top_p",
        "general.sampling.top_k", "general.sampling.min_p",
        "qwen35.block_count", "qwen35.context_length", "qwen35.embedding_length",
        "qwen35.feed_forward_length", "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length", "qwen35.attention.value_length", "qwen35.rope.dimension_count",
        "qwen35.rope.freq_base", "qwen35.ssm.group_count", "qwen35.ssm.state_size", "qwen35.ssm.inner_size",
        "qwen35.ssm.conv_kernel", "qwen35.full_attention_interval", "prism.hadamard.block_size",
        "prism.hadamard.tied_output", "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id", "tokenizer.chat_template")
PROFILE_KEYS = ("qwen35.block_count", "qwen35.embedding_length", "qwen35.feed_forward_length",
                "qwen35.attention.head_count", "qwen35.attention.head_count_kv", "qwen35.ssm.group_count")
# kindi/convert.py:PROFILES, the 27B entry
CONVERT_27B_PROFILE = [64, 5120, 17408, 24, 4, 16]


def parse(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise SystemExit("not a GGUF file")

    def u(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr():
        return f.read(u("<Q")).decode("utf-8", "replace")

    def readval(t):
        if t in S:
            return u(S[t])
        if t == 8:
            return rstr()
        if t == 9:
            at, ln = u("<I"), u("<Q")
            if at == 8:
                for _ in range(ln):
                    f.seek(u("<Q"), 1)
                return "str[%d]" % ln
            if at == 9:
                for _ in range(ln):
                    readval(u("<I"))
                return "arr[%d]" % ln
            f.seek(struct.calcsize(S[at]) * ln, 1)
            return "%s[%d]" % (S[at].strip("<"), ln)
        raise SystemExit("unknown value type %r" % t)

    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    kv, arrays = {}, {}
    for _ in range(n_kv):
        key, t = rstr(), u("<I")
        val = readval(t)
        if isinstance(val, str) and val.startswith(("str[", "arr[")):
            arrays[key] = val
        if key in KEEP:
            kv[key] = val

    types, names = collections.Counter(), []
    for _ in range(n_tensors):
        name = rstr()
        n_dims = u("<I")
        for _ in range(n_dims):
            u("<Q")                                   # shape, not needed for the layout question
        types[u("<I")] += 1
        u("<Q")                                       # data offset
        names.append(name)

    template = kv.get("tokenizer.chat_template") or ""
    prefixes = collections.Counter(n.split(".")[0] for n in names)
    vocab = int(arrays.get("tokenizer.ggml.tokens", "str[0]")[4:-1])
    profile = [kv.get(k) for k in PROFILE_KEYS]
    return {
        "source": {"repo": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF",
                   "file": "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf",
                   "range": "bytes=0-12582911",
                   "bytes_read": 12582912,
                   "http_status": 206,
                   "size_bytes": 5993834848,
                   "hf_sha256": "929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe"},
        "header": {"gguf_version": version, "kv_count": n_kv, "tensor_count": n_tensors},
        "kv": kv,
        "arrays": arrays,
        "vocab": vocab,
        "tensors": {"count": n_tensors,
                    "types": {str(k): v for k, v in sorted(types.items())},
                    "type_names": {str(k): TYPE_NAMES.get(k, "type-%d" % k) for k in sorted(types)},
                    "name_prefixes": dict(sorted(prefixes.items())),
                    "non_block_names": [n for n in names if not n.startswith("blk.")]},
        "chat_template": {"chars": len(template),
                          "has_reasoning_effort": "reasoning_effort" in template,
                          "mentions_xhigh": "xhigh" in template,
                          "mentions_medium": "medium" in template},
        "profile_tuple": profile,
        "convert_profile_match": profile == CONVERT_27B_PROFILE,
    }


if __name__ == "__main__":
    json.dump(parse(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")

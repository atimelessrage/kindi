// SPDX-License-Identifier: AGPL-3.0-only
// Minimal greedy Dory reference harness for the D0 golden audit.
//
// Built from the PR-30306 llama.cpp tree. It is intentionally not part of Kindi's runtime: it pins
// prompt token IDs, generated token IDs and decoded text for one prompt at one dory.num_loops value.

#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf -p PROMPT [-n 32] [-k 2] [-t 6] [-ngl 0]\n"
        "  -k 0 keeps the GGUF dory.num_loops value (normally 2)\n", argv0);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static void print_ids(const char * name, const std::vector<llama_token> & ids) {
    std::printf("\"%s\":[", name);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? "," : "", ids[i]);
    }
    std::printf("]");
}

static bool tokenize(const llama_vocab * vocab, const std::string & text, std::vector<llama_token> & out) {
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, true, true);
    if (n == 0) {
        return true;
    }
    if (n < 0) {
        n = -n;
    }
    out.resize(n);
    int got = llama_tokenize(vocab, text.data(), (int) text.size(), out.data(), n, true, true);
    if (got < 0) {
        return false;
    }
    out.resize(got);
    return true;
}

static bool decode_prompt(llama_context * ctx, llama_batch_ext * batch, const std::vector<llama_token> & toks) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < toks.size(); ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, toks[i]);
        if (idx < 0) {
            return false;
        }
        llama_pos pos = (llama_pos) i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
            return false;
        }
    }
    if (!toks.empty() && !llama_batch_ext_set_output_logits(batch, (int) toks.size() - 1, true)) {
        return false;
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
}

static bool decode_one(llama_context * ctx, llama_batch_ext * batch, llama_token token, llama_pos pos) {
    llama_batch_ext_clear(batch);
    int idx = llama_batch_ext_add_token(batch, 0, token);
    if (idx < 0) {
        return false;
    }
    if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
        return false;
    }
    if (!llama_batch_ext_set_output_logits(batch, idx, true)) {
        return false;
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt;
    int n_predict = 32;
    int k = 2;
    int threads = 6;
    int ngl = 0;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", opt);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "-m") == 0) model_path = need("-m");
        else if (std::strcmp(argv[i], "-p") == 0) prompt = need("-p");
        else if (std::strcmp(argv[i], "-n") == 0) n_predict = std::atoi(need("-n"));
        else if (std::strcmp(argv[i], "-k") == 0) k = std::atoi(need("-k"));
        else if (std::strcmp(argv[i], "-t") == 0) threads = std::atoi(need("-t"));
        else if (std::strcmp(argv[i], "-ngl") == 0) ngl = std::atoi(need("-ngl"));
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty() || prompt.empty() || n_predict <= 0 || threads <= 0) {
        usage(argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model_kv_override kv{};
    if (k > 0) {
        std::strncpy(kv.key, "dory.num_loops", sizeof(kv.key) - 1);
        kv.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kv.val_i64 = k;
        mp.kv_overrides = &kv;
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_ids;
    if (!tokenize(vocab, prompt, prompt_ids) || prompt_ids.empty()) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    if ((int) prompt_ids.size() >= (int) cp.n_ctx - n_predict) {
        std::fprintf(stderr, "prompt too long\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    auto sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (!decode_prompt(ctx, batch, prompt_ids)) {
        std::fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    std::vector<llama_token> generated;
    std::string text;
    llama_pos pos = (llama_pos) prompt_ids.size();
    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        generated.push_back(id);
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::fprintf(stderr, "token_to_piece failed\n");
            return 1;
        }
        text.append(buf, n);
        if (i + 1 < n_predict && !decode_one(ctx, batch, id, pos)) {
            std::fprintf(stderr, "decode failed at token %d\n", i);
            return 1;
        }
        ++pos;
    }

    std::printf("{");
    std::printf("\"model\":\"%s\",", json_escape(model_path).c_str());
    std::printf("\"prompt\":\"%s\",", json_escape(prompt).c_str());
    std::printf("\"k\":%d,", k);
    std::printf("\"n_predict\":%d,", n_predict);
    std::printf("\"threads\":%d,", threads);
    std::printf("\"ngl\":%d,", ngl);
    print_ids("prompt_ids", prompt_ids);
    std::printf(",");
    print_ids("generated_ids", generated);
    std::printf(",\"generated_text\":\"%s\"}\n", json_escape(text).c_str());

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

#include "generate.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace decode;
constexpr size_t vocab = 8, capacity = 96;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void fails(const std::function<void()>& fn) {
    bool caught = false;
    try { fn(); } catch (const std::runtime_error&) { caught = true; }
    require(caught, "expected rejected input");
}
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return uint16_t(bits >> 16);
}
// Orthogonal tied embeddings, zero attention, and a routed SiLU MLP make the
// greedy successor exactly (token + 1) % 8 with a large margin. No model download
// or mocked forward is needed to select acceptance and rejection boundaries.
void fixture(const fs::path& path) {
    fs::create_directories(path);
    Json config{{"model_type", "qwen2"}, {"tie_word_embeddings", true}, {"hidden_act", "silu"},
        {"hidden_size", vocab}, {"intermediate_size", vocab}, {"num_hidden_layers", 1},
        {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"vocab_size", vocab},
        {"max_position_embeddings", capacity}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000}};
    std::ofstream(path / "config.json") << config.dump();
    std::map<std::string, std::vector<uint16_t>> tensors;
    Json header;
    auto add = [&](const std::string& name, std::vector<size_t> shape, float value = 0.0f) -> std::vector<uint16_t>& {
        size_t count = 1; for (size_t n : shape) count *= n;
        auto& data = tensors[name]; data.assign(count, bf16(value));
        header[name] = {{"dtype", "BF16"}, {"shape", shape}};
        return data;
    };
    auto& embedding = add("model.embed_tokens.weight", {vocab, vocab});
    for (size_t i = 0; i < vocab; ++i) embedding[i * vocab + i] = bf16(1);
    add("model.norm.weight", {vocab}, 1);
    const std::string base = "model.layers.0.";
    add(base + "input_layernorm.weight", {vocab}, 1);
    add(base + "post_attention_layernorm.weight", {vocab}, 1);
    for (const char* name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        bool kv = std::string(name) == "k_proj" || std::string(name) == "v_proj";
        size_t rows = kv ? vocab / 2 : vocab;
        add(base + "self_attn." + name + ".weight", {rows, vocab});
        if (std::string(name) != "o_proj") add(base + "self_attn." + name + ".bias", {rows});
    }
    for (const char* name : {"gate_proj", "up_proj"}) {
        auto& data = add(base + "mlp." + name + ".weight", {vocab, vocab});
        for (size_t i = 0; i < vocab; ++i) data[i * vocab + i] = bf16(1);
    }
    auto& down = add(base + "mlp.down_proj.weight", {vocab, vocab});
    for (size_t i = 0; i < vocab; ++i) down[((i + 1) % vocab) * vocab + i] = bf16(1);
    uint64_t offset = 0;
    for (const auto& tensor : tensors) {
        uint64_t bytes = tensor.second.size() * sizeof(uint16_t);
        header[tensor.first]["data_offsets"] = {offset, offset + bytes}; offset += bytes;
    }
    std::string encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
    std::ofstream out(path / "model.safetensors", std::ios::binary);
    uint64_t size = encoded.size(); out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(encoded.data(), encoded.size());
    for (const auto& tensor : tensors)
        out.write(reinterpret_cast<const char*>(tensor.second.data()), tensor.second.size() * sizeof(uint16_t));
    require(bool(out), "write generation fixture");
}
Generation compare(Engine& plain, Engine& lookup, const std::vector<int>& prompt,
                   const std::vector<int>& stops, const std::vector<int>& expected,
                   size_t steps = 16) {
    auto single = generate(plain, prompt, steps, {0, 4, 1}, nullptr, stops);
    auto speculative = generate(lookup, prompt, steps, {4, 4, 8}, nullptr, stops);
    require(single.tokens == expected, "synthetic greedy trajectory differs");
    require(speculative.tokens == single.tokens, "stop-truncated lookup differs from greedy");
    require(single.stopped && speculative.stopped, "missing stop finish reason");
    require(plain.position() == prompt.size() + expected.size() - 1, "ordinary consumed terminal token");
    require(lookup.position() == plain.position(), "speculative terminal cache was not rewound");
    return speculative;
}
void stop_tests(const fs::path& model) {
    for (CacheType cache : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered}) {
        EngineOptions options; options.cache_type = cache; options.strict_affinity = false;
        Engine plain(model.string(), Kernel::scalar, 1, capacity, options);
        Engine lookup(model.string(), Kernel::scalar, 1, capacity, options);
        const std::vector<int> repeat{0, 1, 2, 3, 4, 5, 6, 7, 0};
        auto first = compare(plain, lookup, repeat, {1}, {1});
        require(first.verification_batches == 0 && first.forward_tokens == 0, "stop before draft still forwarded");
        auto draft_first = compare(plain, lookup, repeat, {2}, {1, 2});
        require(draft_first.accepted == 1 && draft_first.drafted == 4, "first draft stop not exercised");
        auto middle = compare(plain, lookup, repeat, {6, 3}, {1, 2, 3});
        require(middle.accepted == 2 && middle.drafted == 4, "interior draft stop not exercised");
        auto last = compare(plain, lookup, repeat, {5}, {1, 2, 3, 4, 5});
        require(last.accepted == 4 && last.verification_batches == 1, "last draft stop not exercised");
        auto bonus = compare(plain, lookup, repeat, {6}, {1, 2, 3, 4, 5, 6});
        require(bonus.accepted == 4 && bonus.verification_batches == 1, "bonus stop not exercised");
        auto mismatch = compare(plain, lookup, {0, 1, 2, 7, 5, 0}, {3}, {1, 2, 3});
        require(mismatch.accepted == 1 && mismatch.drafted == 4, "partial acceptance mismatch not exercised");
        auto rejected = compare(plain, lookup, {0, 1, 7, 6, 0}, {2}, {1, 2});
        require(rejected.accepted == 0 && rejected.drafted > 0, "first mismatch stop not exercised");
        // An EOS in a rejected draft is not an output EOS.
        compare(plain, lookup, {0, 1, 2, 7, 5, 0}, {7}, {1, 2, 3, 4, 5, 6, 7});
        compare(plain, lookup, repeat, {3}, {1, 2, 3}, 3);
        // Stop and rollback while the centered cache crosses its frozen-prefix boundary.
        auto long_prompt = std::vector<int>(53, 7);
        long_prompt.insert(long_prompt.end(), repeat.begin(), repeat.end());
        compare(plain, lookup, long_prompt, {3}, {1, 2, 3});
        // Reuse each stopped engine: no suffix or stale stop state survives reset.
        auto fixed = generate(plain, repeat, 20, {0, 4, 1});
        auto fixed_lookup = generate(lookup, repeat, 20, {4, 4, 8});
        std::vector<int> expected;
        for (size_t i = 0; i < 20; ++i) expected.push_back(int((i + 1) % vocab));
        require(fixed.tokens == expected && fixed_lookup.tokens == expected, "fixed-token behavior changed");
        require(!fixed.stopped && !fixed_lookup.stopped, "implicit EOS changed fixed-token finish reason");
        require(plain.position() == repeat.size() + 19 && lookup.position() == plain.position(), "fixed-token final position");
    }
}
void capacity_tests(const fs::path& model) {
    EngineOptions options; options.strict_affinity = false;
    const std::vector<int> prompt{0, 1, 2, 3, 4, 5, 6, 7, 0};
    Engine full(model.string(), Kernel::scalar, 1, capacity, options);
    std::vector<int> full_prompt(capacity, 0);
    for (size_t drafts : {size_t(0), size_t(4)}) {
        auto one = generate(full, full_prompt, 1, {drafts, 4, 8}, nullptr, {1});
        require(one.tokens == std::vector<int>{1} && one.stopped, "full prompt cannot select final EOS");
        require(full.position() == capacity, "one token needs an unnecessary KV slot");
        auto zero = generate(full, full_prompt, 0, {drafts, 4, 8}, nullptr, {1});
        require(zero.tokens.empty() && !zero.stopped && full.position() == capacity, "zero-step generation changed");
        fails([&] { generate(full, full_prompt, 2, {drafts, 4, 8}); });
        Engine exact(model.string(), Kernel::scalar, 1, prompt.size() + 15, options);
        auto fixed = generate(exact, prompt, 16, {drafts, 4, 8});
        require(fixed.tokens.size() == 16 && exact.position() == prompt.size() + 15, "exact decode capacity failed");
        for (size_t count : {size_t(2), size_t(3)}) {
            Engine short_run(model.string(), Kernel::scalar, 1, prompt.size() + count - 1, options);
            std::vector<int> expected{1, 2};
            if (count == 3) expected.push_back(3);
            auto short_result = generate(short_run, prompt, count, {drafts, 4, 8});
            require(short_result.tokens == expected && !short_result.stopped, "short output budget changed");
            require(short_run.position() == prompt.size() + count - 1, "short output budget consumed final token");
        }
    }
    // Early EOS may fit even when the maximum output budget does not. A
    // speculative batch must not consume unavailable slots past that EOS.
    for (size_t stop : {size_t(2), size_t(3), size_t(5)}) {
        Engine plain(model.string(), Kernel::scalar, 1, prompt.size() + stop - 1, options);
        Engine lookup(model.string(), Kernel::scalar, 1, prompt.size() + stop - 1, options);
        std::vector<int> expected;
        for (size_t token = 1; token <= stop; ++token) expected.push_back(int(token));
        compare(plain, lookup, prompt, {int(stop)}, expected, 16);
    }
    full.reset(); full.step(0, false);
    for (int invalid : {-1, int(vocab)}) {
        fails([&] { generate(full, prompt, 4, {}, nullptr, {invalid}); });
        require(full.position() == 1, "invalid stop token changed cache state");
    }
}
void binding_tests(const fs::path& model) {
    cpu_set_t original;
    require(sched_getaffinity(0, sizeof(original), &original) == 0, "read original affinity");
    auto restored = [&] {
        cpu_set_t current;
        require(sched_getaffinity(0, sizeof(current), &current) == 0, "read restored affinity");
        require(CPU_EQUAL(&original, &current), "request did not restore exact caller affinity");
    };
    for (bool strict : {true, false}) {
        EngineOptions options; options.strict_affinity = strict;
        Engine engine(model.string(), Kernel::scalar, 2, capacity, options);
        restored();
        int cpu = strict ? engine.metadata().at("cpu_set").at(0).get<int>() : -1;
        {
            CpuBinding request(cpu);
            auto generated = generate(engine, {0, 1, 2, 3, 4, 5, 6, 7, 0}, 16, {4, 4, 8}, nullptr, {3});
            require(generated.tokens == std::vector<int>({1, 2, 3}), "bound generation differs");
            cpu_set_t current;
            require(sched_getaffinity(0, sizeof(current), &current) == 0, "read request affinity");
            require(strict ? CPU_COUNT(&current) == 1 && CPU_ISSET(cpu, &current) : CPU_EQUAL(&original, &current),
                    "nested forwards changed request binding");
        }
        restored();
        fails([&] {
            CpuBinding request(cpu);
            generate(engine, {0}, 4, {}, nullptr, {-1});
        });
        restored();
    }
}
} // namespace
int main() {
    char name[] = "/tmp/cpu-decode-generation-XXXXXX";
    char* directory = mkdtemp(name);
    if (!directory) { std::cerr << "cannot create fixture directory\n"; return 1; }
    fs::path root(directory);
    try {
        fixture(root); stop_tests(root); capacity_tests(root); binding_tests(root);
        fs::remove_all(root);
        std::cout << "generation stop, speculation, capacity, and binding tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root); std::cerr << error.what() << '\n'; return 1;
    }
}

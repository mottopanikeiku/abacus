CC ?= cc
CFLAGS ?= -O3
ifneq (,$(filter $(shell uname -m),aarch64 arm64))
ARCH ?= -mcpu=native
else
ARCH ?= -march=native
endif
# Any dense Qwen3 works: make model MODEL=Qwen3-4B writes Qwen3-4B.bin.
MODEL ?= Qwen3-0.6B
BIN = $(if $(filter Qwen3-0.6B,$(MODEL)),model.bin,$(MODEL).bin)
REF = $(if $(filter Qwen3-0.6B,$(MODEL)),ref,ref-$(MODEL))

abacus: src/*.c src/*.h
	$(CC) $(CFLAGS) $(ARCH) -std=c11 -D_GNU_SOURCE -pthread -Wall -Wextra -o $@ src/*.c -lm

# Download the model from Hugging Face (Qwen3-0.6B is 1.5 GB) and convert it.
$(BIN):
	python3 tools/download.py Qwen/$(MODEL) models/$(MODEL)
	python3 tools/export.py models/$(MODEL) $@

model: $(BIN)

# Reference logits from Hugging Face transformers (needs uv), then the check.
$(REF)/ref.json:
	uv run tools/reference.py models/$(MODEL) $(REF)

check: abacus $(BIN) $(REF)/ref.json
	uv run tools/check.py ./abacus $(BIN) $(REF)

clean:
	rm -f abacus

.PHONY: model check clean

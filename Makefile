CC ?= cc
CFLAGS ?= -O3
ifneq (,$(filter $(shell uname -m),aarch64 arm64))
ARCH ?= -mcpu=native
else
ARCH ?= -march=native
endif
MODEL_DIR ?= models/Qwen3-0.6B
HF = https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main

abacus: src/*.c src/*.h
	$(CC) $(CFLAGS) $(ARCH) -std=c11 -D_GNU_SOURCE -Wall -Wextra -o $@ src/*.c -lm

# Download Qwen3-0.6B (1.5 GB) and convert it.
model.bin:
	mkdir -p $(MODEL_DIR)
	for f in config.json tokenizer.json model.safetensors; do \
		test -f $(MODEL_DIR)/$$f || curl -fL --progress-bar -o $(MODEL_DIR)/$$f $(HF)/$$f || exit 1; done
	python3 tools/export.py $(MODEL_DIR) $@

model: model.bin

# Reference logits from Hugging Face transformers (needs uv), then the check.
ref/ref.json:
	uv run tools/reference.py $(MODEL_DIR) ref

check: abacus model.bin ref/ref.json
	uv run tools/check.py ./abacus model.bin ref

clean:
	rm -f abacus

.PHONY: model check clean

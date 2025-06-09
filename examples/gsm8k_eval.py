# %% [markdown]
# # GSM8K Evaluation Notebook
#
# This notebook evaluates Gemma JAX on the GSM8K mathematical reasoning dataset.

# %% [markdown]
# ### Imports and Environment Setup
#
# Conditionally installs dependencies and sets up the environment based on whether it's running locally or in Google Colab.

# %%

try:
  import google.colab

  IN_COLAB = True
except ImportError:
  IN_COLAB = False

if IN_COLAB:
  from google.colab import drive

  drive.mount("/content/drive")

  import os

  os.chdir("/content/drive/My Drive/gemma-jax")
  print(f"Running in Google Colab. Current directory: {os.getcwd()}")
else:
  import os
  from pathlib import Path

  home_dir = Path.home()
  os.chdir(home_dir / "docs" / "gemma-jax")
  print(f"Running locally. Current directory: {Path.cwd()}")

# %% [markdown]
# ### Installation
#
# Install required Python packages for TPU support and dataset management. The command below quietly installs dependencies suitable for TPU-based inference.

# %%
# !pip install -e . --quiet

# %% [markdown]
# ### Configuration Defaults
#
# Set the default paths and parameters. Adjust the paths to reflect your actual filesystem setup for the tokenizer and model checkpoints.

# %%
from pathlib import Path

root_dir = Path.cwd()
checkpoint_path = root_dir / "4b"  # Absolute path to the Gemma model checkpoint
tokenizer_path = root_dir / "tokenizer.model"  # Absolute path to SentencePiece tokenizer

assert tokenizer_path.exists(), f"Tokenizer path {tokenizer_path} does not exist."
assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist."

print(f"Tokenizer path: {tokenizer_path}")
print(f"Checkpoint path: {checkpoint_path}")

# %% [markdown]
# ### Core Package Imports
#
# Import essential components from `gemma-jax` required for model configuration, loading, inference, and text processing.

# %%
import time
from functools import partial
import jax
from jax import numpy as jnp
from gemma_jax.core.weights import create_gemma3_config, create_device_mesh, load_model
from gemma_jax.core.model import forward_fn, setup_scan_fn, scan_generate_step
from gemma_jax.core.cache import KVCache, LayoutType, SEQUENCE_HEADS, init_cache, layout_map
from gemma_jax.core.rope import load_rope_cache
from gemma_jax.core.sp_tokenizer import SentencePieceTokenizer, process_and_pad_inputs, encode_text, decode_tokens, format_prompt
from gemma_jax.core.inference import greedy_sample


# %% [markdown]
# ### Model Configuration
#
# Define model hyperparameters and inference settings, such as cache size, sequence lengths, batch size, and data types. Adjust these according to your computational resources and experimental needs.

# %%
model_size = 4  # Model scale (e.g., 4 for 4B parameters)
cache_length = 4096  # Length of KV-cache
padded_input_size = 1024  # Maximum input sequence length
window_size = 1024  # Attention window size

batch_size = 2  # Adjust according to your TPU/ CPU setup
generate_steps = 1024  # Number of tokens generated after prefill
save_every = 100  # Save evaluation results every 100 examples
max_num_examples = 120  # Limit the number of examples for evaluation
print_every = 20  # Print every 20 examples
save_path = "results.json"  # Path to save evaluation results
dtype_str = "bfloat16"
model_dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[dtype_str]

# XLA # TODO: if needed, update mem_fraction
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


# %% [markdown]
# ### Initialization and Model Loading
#
# This step includes initializing the device mesh, loading the Gemma model checkpoint, creating RoPE caches, initializing KV caches, and loading the tokenizer. It dynamically adjusts settings based on the detected hardware (CPU or TPU).

# %%
start_setup = time.time()

if jax.devices()[0].device_kind == "cpu":
  print("Using CPU device settings.")
  cache_length, padded_input_size, window_size = 1024, 128, 128
  batch_size, generate_steps, save_every, max_num_examples = 1, 4, 100, 10
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"


mesh = create_device_mesh()  # Adjust the mesh shape based on the number of devices and TPU arch

config = create_gemma3_config(
    model_size=model_size,
    batch_size=batch_size,
    padded_input_size=padded_input_size,
    cache_length=cache_length,
    window_size=window_size,
    generate_steps=generate_steps,
)

rope_cache = load_rope_cache(mesh, config)
cache = init_cache(mesh=mesh, config=config, dtype=jnp.bfloat16, kind=SEQUENCE_HEADS, layout_map=layout_map)
tokenizer = SentencePieceTokenizer(tokenizer_path)

model = load_model(checkpoint_path, mesh, config, dtype=model_dtype)
print(f"Setup complete in {time.time() - start_setup:.2f}s")


# %% [markdown]
# ## Evaluation Constants
#
# Define constants for conversational roles and tokenizer special tokens for consistent handling of prompts and model outputs.

# %%
ROLE_USER, ROLE_MODEL, ROLE_SYSTEM = 0, 1, 2
ROLE_STR = {ROLE_USER: "user", ROLE_MODEL: "model", ROLE_SYSTEM: "system"}

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2

# %% [markdown]
# ## Prompt Templates
#
# Prompts for GSM8K evaluation are adapted from the original GSM8K benchmark provided by Google DeepMind.
#
# Original notebook: [gsm8k_eval.ipynb](https://github.com/google-deepmind/gemma/blob/2a162e21be390aa0ec635deb7176fb64fb1868b1/colabs/old/gsm8k_eval.ipynb)
#
# (See commit: 2a162e21be390aa0ec635deb7176fb64fb1868b1)

# %%
from gemma_jax.core.gsm8k_eval import PREAMBLE, EXTRA_3_SHOTS, FEWSHOT, PROMPT as EIGHT_SHOT_PROMPT

# --- System prompt ---
PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

FEWSHOT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
"""


# %% [markdown]
# ## Helper Functions
#
# Utility functions for parsing and extracting numerical results from model outputs.

# %%
import re


def find_numbers(x: str) -> list[str]:
  return re.compile(r"-?[\d,]*\.?\d+").findall(x)


def find_number(x: str, answer_delimiter: str = "The answer is") -> str:
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ""


def normalize_answer(ans: str) -> str:
  ans = ans.strip().replace(",", "")
  ans = re.sub(r"[^\d\.\-]+$", "", ans)
  return ans


# --- Utilities ----

_NUM_RE = r"-?[\d,]*\.?\d+"


def _strip(s: str) -> str:
  return s.strip(" \n\t")


def _visible_segment(raw: str) -> str:
  """
  Pull out the *model-visible* text from the raw dump emitted by
  `chat_once_batched`.  Supports both:
       “<start_of_turn>model … <end_of_turn>”
       “user\n … model\n …”
  """
  # 1) Try the explicit tag format first
  m = re.search(r"<start_of_turn>model(.*?)<end_of_turn>", raw, flags=re.S)
  if m:
    return _strip(m.group(1))

  # 2) Fallback: last “…\nmodel\n” chunk
  if "\nmodel" in raw:
    # Split by the LAST occurrence to avoid grabbing few-shot exemplars
    head, _, tail = raw.rpartition("\nmodel")
    return _strip(tail)

  # 3) Nothing matched -return full string (best effort)
  return _strip(raw)


def _last_answer_block(visible: str) -> str:
  """
  Given the visible text, grab the **final** answer paragraph:

      … Q: <few-shot #N>
         A: <few-shot #N answer>
      Q: <real question>
      A: <real answer>

  We split on *lines* that start with 'Q:' or 'A:' and keep the
  trailing answer.
  """
  # Normalise line endings & get rid of leading markdown bullets
  lines = [l.lstrip("* ").rstrip() for l in visible.replace("\r", "").split("\n")]

  # Walk backwards to find the last line that *begins* with 'A:'
  for i in range(len(lines) - 1, -1, -1):
    if lines[i].startswith(("A:", "Answer:", "**Answer:**")):
      # Everything FROM that line onwards is the answer block
      return "\n".join(lines[i:])

  # Fallback -no marker, keep entire visible
  return visible


def _extract_number(ans_block: str) -> str:
  """
  Pull the numeric answer out of the final answer block.
  Priority:
      1.  “The answer is <num>”
      2.  “Answer: <blah> <num>”
      3.  Last number in the block
  TODO:
  handle model outputs with "commas" or other variations, for example, 'model_answer': 99,076.92 ,   'predicted': '99076.92', is a valid answer, but missed by the parser.

  """
  # 1) “The answer is<num>”
  m = re.search(r"The answer is\s*(%s)" % _NUM_RE, ans_block, flags=re.I)
  if m:
    return _strip(m.group(1)).replace(",", "")

  # 2) “Answer:” / “**Answer:**”
  m = re.search(r"Answer[:\s\*]*.*?(%s)" % _NUM_RE, ans_block, flags=re.I | re.S)
  if m:
    return _strip(m.group(1)).replace(",", "")

  # 3) Anything else -last number in the block
  nums = re.findall(_NUM_RE, ans_block)
  return nums[-1].replace(",", "") if nums else ""


def _parse_reply(raw_reply: str) -> tuple[str, str]:
  """
  Returns (visible_str, numeric_prediction)
  """
  visible = _visible_segment(raw_reply)
  answer_block = _last_answer_block(visible)
  num = _extract_number(answer_block)
  return visible, num


# %% [markdown]
# ## Evaluation Setup

# Create text preprocessing and inference functions for evaluation. These functions will be passed to the main evaluation loop.

# %% [markdown]
# ### Input Preparation and Prefill
#
# Prepare batched inputs by tokenizing, padding, and processing prompts for prefill and autoregressive generation steps.

# %%
process_inputs = partial(
    process_and_pad_inputs, max_sequence_length=padded_input_size, cache_len=cache_length, tokenizer=tokenizer
)

prefill_model = partial(
    forward_fn, write_index=0, model=model, cache=cache, rope_cache=rope_cache, config=config, cache_layout=SEQUENCE_HEADS
)

generate_step = partial(scan_generate_step, model=model, rope_cache=rope_cache, config=config, cache_layout=SEQUENCE_HEADS)

# %% [markdown]
# ## Evaluation Procedure
#
# The key functions imported from `conversation_state.py` include:
# - `create_empty_state_batched`: Initializes the state object for maintaining conversation history.
# - `chat`: Orchestrates the inference loop for generating responses.

# %%

import json
import datasets
from tqdm import tqdm
from gemma_jax.core.conversation_state import create_empty_state_batched, chat

QUESTION_TEMPLATE = "\nQ: {question}\nA:"


def evaluate_gsm8k_batched(
    batch_size: int = 4,
    save_path: str = save_path,
    generate_steps: int = 256,
    verbose: bool = True,
    save_every: int = 50,
    print_every: int = 20,
    max_num_examples: int | None = None,
    max_turns: int = 10,
    tokenizer=tokenizer,
    config=config,
    benchmark=FEWSHOT,
):

  #  0.  Load data & assemble prompts
  gsm8k = datasets.load_dataset("gsm8k", "main")["test"]
  if max_num_examples is not None:
    gsm8k = gsm8k.select(range(max_num_examples))

  prompts = [format_prompt(f"{PREAMBLE}\n{FEWSHOT}{QUESTION_TEMPLATE.format(question=ex['question'])}") for ex in gsm8k]
  ground_truths = [ex["answer"] for ex in gsm8k]
  questions = [ex["question"] for ex in gsm8k]
  num_examples = len(prompts)
  print(f"Number of examples: {num_examples}")

  #  1.  Initialise chat state
  conv_state = create_empty_state_batched(
      batch_size=config.batch_size,
      cache_length=config.cache_length,
      max_turns=max_turns,
      with_trace=False,
      pad_id=PAD_ID,
  )

  # 2. Set up the initial conversation state
  chat_partial = partial(
      chat,
      prefill_model,
      process_inputs,
      setup_scan_fn,
      generate_step,
      tokenizer,
      config,
  )

  # Book-keeping accumulators
  results: list[dict] = []
  correct = 0
  truncations = 0

  print(f"Evaluating {num_examples} examples " f"in batches of {batch_size} …")

  #  3.  Main loop -batched inference + parsing
  for i in tqdm(range(0, num_examples, batch_size), desc="GSM8K batched"):

    batch_prompts = prompts[i : i + batch_size]

    # Guard: last batch may be smaller than `batch_size`
    cur_bs = len(batch_prompts)
    if cur_bs < batch_size:
      # Skip if not multiple of batch size
      continue
    else:
      conv_state_batch = conv_state

    _, batch_replies = chat_partial(
        conv_state_batch,
        batch_prompts,
        generate_steps=generate_steps,
        role=ROLE_USER,
    )

    # Vectorised parsing (no JIT -but consistent shape for later)
    batch_vis_preds = list(map(_parse_reply, batch_replies))

    #  4.  Per-example metrics & logging
    for j, ((visible, pred), question, gt_raw) in enumerate(
        zip(
            batch_vis_preds,
            questions[i : i + cur_bs],
            ground_truths[i : i + cur_bs],
        )
    ):
      idx = i + j
      truth = _extract_number(gt_raw)
      is_correct = pred == truth

      # Simple truncation heuristic -no answer block found
      is_truncated = pred == ""

      if is_correct:
        correct += 1
      if is_truncated:
        truncations += 1

      results.append(
          {
              "idx": idx,
              "question": question,
              "ground_truth": truth,
              "model_answer": visible,
              "predicted": pred,
              "is_correct": is_correct,
              "truncated": is_truncated,
              "raw": batch_replies[j],
          }
      )

      # TODO: print all incorrect answers by addin: or not is_correct):
      if verbose and ((idx + 1) % print_every == 0):
        running_acc = correct / (idx + 1)
        running_trunc = truncations / (idx + 1)
        print(
            f"\n--- Example{idx+1}/{num_examples}"
            f"\nQ: {question}"
            f"\nGT: {truth}\nPred: {pred} | Correct: {is_correct}"
            f"\nRunning Acc: {running_acc:.2%} |Trunc rate: {running_trunc:.2%}"
        )

      # periodic checkpoint
      if (idx + 1) % save_every == 0:
        with open(save_path, "w") as fp:
          json.dump(results, fp, indent=2)
        if verbose:
          print(f"[checkpoint] saved{idx+1} examples → {save_path}")

  #  5.  Final stats + save
  accuracy = correct / len(results) if len(results) > 0 else 0
  print(f"\nGSM8K accuracy: {accuracy:.2%} " f"({correct}/{len(results)})  |truncations: {truncations}")

  with open(save_path, "w") as fp:
    json.dump(results, fp, indent=2)
  print(f"[final] wrote results to {save_path}")

  return accuracy, results


# %% [markdown]
# ## Evaluation Execution
#
# Execute the cell below to run the full evaluation. The runtime can vary significantly based on hardware capabilities.

# %%
accuracy, results = evaluate_gsm8k_batched(
    batch_size=batch_size,
    generate_steps=generate_steps,
    save_every=save_every,
    print_every=20,
    verbose=True,
    max_num_examples=max_num_examples,
)

# %% [markdown]
# ## Results Summary
#
# The [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786v1) presents the following GSM8K results:
# - Instruction-finetuned 4B model achieves **89.2%** (8-shot, Chain-of-Thought).
# - Pre-trained 4B model baseline: **38.4%**.
#
# **Our results:**
# - **61.67% accuracy** over 120 examples using the FEWSHOT prompt (no explicit Chain-of-Thought prompting).
#
# Refer to Tables 10 and 18 in the technical report for comprehensive benchmark results.

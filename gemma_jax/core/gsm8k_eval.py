"""
Basic evaluation script for the GSM8K dataset.

This script loads the GSM8K dataset, formats the prompts, and batch evaluates the model's performance on the dataset.
It uses the `chat` function to generate responses and parses the results to extract the answers. Skeleton support for
for 'thinking', 'system' and other control tokens. Results are printed to the console and saved to a JSON file.

GSM8K prompts taken from 'google-deepmind/gemma/tree/main/colabs/old/gsm8k_eval.ipynb'
, see commit: 2a162e21be390aa0ec635deb7176fb64fb1868b1 for original work.
"""

from dataclasses import replace
import datasets
import json
import re
import time
from tqdm import tqdm
from gemma_jax.core.sp_tokenizer import format_prompt
from gemma_jax.core.conversation_state import create_empty_state_batched
from typing import Callable, Any
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct
import json


# --- Constants ---
PAD_ID = 0
EOS_ID = 1
BOS_ID = 2

# --- Role IDs ---
ROLE_USER, ROLE_MODEL, ROLE_SYSTEM = 0, 1, 2
ROLE_STR = {ROLE_USER: "user", ROLE_MODEL: "model", ROLE_SYSTEM: "system"}

# --- Dialogue prompt/ answer wrappers ---
PROMPT_TEMPLATE = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
ANSWER_TEMPLATE = "{}<end_of_turn>"

# --- GSM8K Qustion templates ---
QUESTION_TEMPLATE = "\nQ: {question}\nA:"

# --- GSM8K Prompt Setup ---

PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.

PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""


# Extension of the default 8-shot prompt, page 35 in
# https://arxiv.org/pdf/2201.11903.pdf
# The extension is intended to improve performance on
# more complicated gsm8k examples.

EXTRA_3_SHOTS = """As an expert problem solver solve step by step the following mathematical questions.

Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
A: Here's how to calculate Tina's earnings:

**Regular Time:**
- Hours per shift: 8 hours
- Wage per hour: $18.00
- Regular pay per shift: 8 hours * $18.00/hour = $144.00

**Overtime:**
- Overtime hours per shift: 10 hours - 8 hours = 2 hours
- Overtime pay per hour: $18.00 + ($18.00 / 2) = $27.00
- Overtime pay per shift: 2 hours * $27.00/hour = $54.00

**Total per day:**
- Regular pay + overtime pay: $144.00/shift + $54.00/shift = $198.00/day

**Total for 5 days:**
- 5 days * $198.00/day = $990.00

**Therefore, Tina will make $990.00 in 5 days.** The answer is 990.

Q: Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?
A: ## Ambiguity in the Problem Statement:

There is one main ambiguity in the problem statement:

**Total volume vs. Number of servings:** The statement "18 total cups of this drink" could be interpreted in two ways:
  * **18 cups of the combined volume:** This would mean Abigail used a total of 18 cups of liquid, including both iced tea and lemonade.
  * **18 individual servings:** This would mean Abigail made 18 individual drinks, each containing 1/4 cup of iced tea and 1 1/4 cup of lemonade.

Let us assume the interpretation "18 cups of the combined volume".

## Solution assuming 18 cups of combined volume:

**Step 1: Find the proportion of lemonade in one drink:**

* Lemonade: 1 1/4 cups
* Iced tea: 1/4 cup
* Total: 1 1/4 + 1/4 = 1 1/2 cups
* Lemonade proportion: (1 1/4) / (1 1/2) = 5/6

**Step 2: Calculate the amount of lemonade in the pitcher:**

* Total volume: 18 cups
* Lemonade proportion: 5/6
* Volume of lemonade: 18 * (5/6) = 15 cups

Therefore, there are 15 cups of lemonade in the pitcher. The answer is 15.

Q: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
A: Let us solve it using algebra. Let x be the number of people on the ship the monster ate in the first hundred years.

The number of people on the ship eaten in the second hundred years is 2x, and in the third hundred years is 4x.

Therefore, the total number of people eaten over three hundred years is x + 2x + 4x = 847.

Combining like terms, we get 7x = 847.

Dividing both sides by 7, we find x = 121.

Therefore, there were 121 people on the ship the monster ate in the first hundred years. The answer is 121."""


FEWSHOT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
"""

# Enhanced 3-shot CoT examples with structured reasoning
COT_3SHOT = """\
Q: There are 15 trees in the grove. Grove workers will plant trees today. After planting, there will be 21 trees. How many trees were planted?
A: <thinking>
1. Initial trees: 15
2. Final trees: 21
3. Trees planted = Final - Initial = 21 - 15 = 6
</thinking>
The answer is 6.

Q: A bakery sells 12 cookies per box. If they baked 100 cookies and sold 7 boxes, how many cookies remain?
A: <thinking>
1. Total baked: 100
2. Cookies sold: 7 boxes × 12 cookies/box = 84
3. Remaining cookies = 100 - 84 = 16
</thinking>
The answer is 16.

Q: A classroom has 30 desks. If 4 desks are broken and 3 new ones arrive, how many usable desks are there?
A: <thinking>
1. Initial usable: 30 - 4 = 26
2. After delivery: 26 + 3 = 29
</thinking>
The answer is 29.
"""


@struct.dataclass
class CoTEvalState:
  correct: Array
  total: Array
  cot_hits: Array


def _create_cot_prompt(question: str) -> str:
  return f"{COT_3SHOT}\nQ: {question}\nA: <thinking>\n"


@jax.jit
def _update_eval_state(state: CoTEvalState, batch_correct: Array, batch_cot_hits: Array) -> CoTEvalState:
  return CoTEvalState(
      correct=state.correct + batch_correct.sum(),
      total=state.total + batch_correct.size,
      cot_hits=state.cot_hits + batch_cot_hits.sum(),
  )


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


def advanced_extract_response(text: str) -> dict:
  start = text.rfind("<start_of_turn>model")
  if start != -1:
    text = text[start + len("<start_of_turn>model") :]
  end = text.find("<end_of_turn>")
  if end != -1:
    text = text[:end]
  visible = text.strip()
  thinking = ""
  system = ""
  truncation = False
  if "<thinking>" in visible and "</thinking>" in visible:
    thinking_start = visible.find("<thinking>")
    thinking_end = visible.find("</thinking>")
    thinking = visible[thinking_start + len("<thinking>") : thinking_end].strip()
    visible = (visible[:thinking_start] + visible[thinking_end + len("</thinking>") :]).strip()
  if "<system>" in visible and "</system>" in visible:
    system_start = visible.find("<system>")
    system_end = visible.find("</system>")
    system = visible[system_start + len("<system>") : system_end].strip()
    visible = (visible[:system_start] + visible[system_end + len("</system>") :]).strip()
  if visible.endswith("...") or len(visible) == 0:
    truncation = True
  return {
      "full": text.strip(),
      "visible": visible.strip(),
      "thinking": thinking,
      "system": system,
      "truncated": truncation,
  }


# --- Utilities ----

_NUM_RE = r"-?[\d,]*\.?\d+"


def _strip(s: str) -> str:
  return s.strip(" \n\t")


def _visible_segment(raw: str) -> str:
  """
  Pull out the *model-visible* text from the raw dump emitted by
  `chat_once_batched`.  Supports both:
       "<start_of_turn>model … <end_of_turn>"
       "user\n … model\n …"
  """
  # 1) Try the explicit tag format first
  m = re.search(r"<start_of_turn>model(.*?)<end_of_turn>", raw, flags=re.S)
  if m:
    return _strip(m.group(1))

  # 2) Fallback: last "…\nmodel\n" chunk
  if "\nmodel" in raw:
    # Split by the LAST occurrence to avoid grabbing few-shot exemplars
    head, _, tail = raw.rpartition("\nmodel")
    return _strip(tail)

  # 3) Nothing matched - return full string (best effort)
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

  # Fallback - no marker, keep entire visible
  return visible


def _extract_number(ans_block: str) -> str:
  """
  Pull the numeric answer out of the final answer block.
  Priority:
      1.  "The answer is <num>"
      2.  "Answer: <blah> <num>"
      3.  Last number in the block
  TODO:
  handle model outputs with "commas" or other variations, for example, 'model_answer': 99,076.92 ,   'predicted': '99076.92', is a valid answer, but missed by the parser.

  """
  # 1) "The answer is <num>"
  m = re.search(r"The answer is\s*(%s)" % _NUM_RE, ans_block, flags=re.I)
  if m:
    return _strip(m.group(1)).replace(",", "")

  # 2) "Answer:" / "**Answer:**"
  m = re.search(r"Answer[:\s\*]*.*?(%s)" % _NUM_RE, ans_block, flags=re.I | re.S)
  if m:
    return _strip(m.group(1)).replace(",", "")

  # 3) Anything else - last number in the block
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


# --- Main evaluation loop ----
def evaluate_gsm8k_batched(
    generate_steps: int,
    config: Any,
    max_turns: int = 10,
    with_trace: bool = False,
    save_path: str = "results.json",
    verbose: bool = True,
    save_every: int = 50,
    print_every: int = 20,
    max_num_examples: int | None = None,
    chat_partial: Callable = None,
):
  batch_size = config.batch_size
  cache_length = config.cache_length

  # Timestamp the results file (Unix epoch)
  save_path = f"results_{int(time.time())}.json"

  #  0.  Load data & assemble prompts
  gsm8k = datasets.load_dataset("gsm8k", "main")["test"]
  if max_num_examples is not None:
    gsm8k = gsm8k.select(range(max_num_examples))

  # Select the last 20 examples
  # gsm8k = datasets.load_dataset("gsm8k", "main")["test"]
  # last_20 = gsm8k.select(range(len(gsm8k) - 21, len(gsm8k)))
  # gsm8k = last_20
  # Select examples 100 to 140 (inclusive of 100, exclusive of 140)
  # subset = gsm8k.select(range(100, 140))

  prompts = [
      format_prompt(f"{PREAMBLE}\n{EXTRA_3_SHOTS}{QUESTION_TEMPLATE.format(question=ex['question'])}") for ex in gsm8k
  ]
  ground_truths = [ex["answer"] for ex in gsm8k]
  questions = [ex["question"] for ex in gsm8k]
  num_examples = len(prompts)
  print(f"Number of examples: {num_examples}")

  #  1.  Initialise chat state
  # conv_state = create_empty_state_batched(batch_size, state.max_tok, state.max_turn)

  conv_state = create_empty_state_batched(
      batch_size=config.batch_size,
      cache_length=config.cache_length,
      max_turns=max_turns,
      with_trace=with_trace,
      pad_id=PAD_ID,
  )

  #  2.  Book-keeping accumulators
  results: list[dict] = []
  correct = 0
  truncations = 0

  print(f"Evaluating {num_examples} examples " f"in batches of {batch_size} …")

  #  3.  Main loop - batched inference + parsing
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

    # Vectorised parsing (no JIT - but consistent shape for later)
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

      # Simple truncation heuristic - no answer block found
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

      # periodic console output
      if verbose and ((idx + 1) % print_every == 0):
        running_acc = correct / (idx + 1)
        running_trunc = truncations / (idx + 1)
        print(
            f"\n--- Example {idx+1}/{num_examples}"
            f"\nQ: {question}"
            f"\nGT: {truth}\nPred: {pred} | Correct: {is_correct}"
            f"\nRunning Acc: {running_acc:.2%} | Trunc rate: {running_trunc:.2%}"
        )

      # periodic checkpoint
      if (idx + 1) % save_every == 0:
        with open(save_path, "w") as fp:
          json.dump(results, fp, indent=2)
        if verbose:
          print(f"[checkpoint] saved {idx+1} examples → {save_path}")

  #  5.  Final stats + save
  accuracy = correct / len(results)
  print(f"\nGSM8K accuracy: {accuracy:.2%} " f"({correct}/{len(results)})  | truncations: {truncations}")

  with open(save_path, "w") as fp:
    json.dump(results, fp, indent=2)
  print(f"[final] wrote results to {save_path}")

  return accuracy, results

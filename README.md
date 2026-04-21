# esme-base-app

A hands-on course that takes you from a single LLM API call to a fully observable, tool-using AI agent — step by step.  
Each numbered Python script is a self-contained lesson that builds on the previous one.

---

## Table of Contents

1. [What You Will Learn](#what-you-will-learn)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Course Lessons](#course-lessons)
   - [Lesson 1 — Base LLM Call](#lesson-1--base-llm-call)
   - [Lesson 2 — Logging with Langfuse](#lesson-2--logging-with-langfuse)
   - [Lesson 3 — Multi-step Agent](#lesson-3--multi-step-agent)
   - [Lesson 4 — Datasets & Experiments](#lesson-4--datasets--experiments)
   - [Lesson 5 — LLM as a Judge](#lesson-5--llm-as-a-judge)
   - [Lesson 6 — Tool Use with smolagents](#lesson-6--tool-use-with-smolagents)
5. [How It All Fits Together](#how-it-all-fits-together)
6. [Tips & Best Practices](#tips--best-practices)

---

## What You Will Learn

| Skill | Covered in |
|---|---|
| Call an LLM via the Groq API | Lesson 1 |
| Trace and observe LLM calls with Langfuse | Lesson 2 |
| Build a multi-step reasoning pipeline | Lesson 3 |
| Create evaluation datasets and run experiments | Lesson 4 |
| Use another LLM to judge the quality of outputs | Lesson 5 |
| Give an agent real tools it can call at runtime | Lesson 6 |

---

## Prerequisites

- Python 3.12+
- A [Groq](https://console.groq.com/) account and API key
- A [Langfuse](https://cloud.langfuse.com/) account (free tier is enough) with a project's public/secret keys
- Basic familiarity with Python functions and `pip`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Atuu4012/esme-base-app.git
cd esme-base-app

# 2. Install dependencies (uv is recommended, plain pip also works)
pip install .
# or: uv sync

# 3. Copy the example env file and fill in your keys
cp .env.example .env
```

Open `.env` and replace the placeholder values:

```env
GOOGLE_API_KEY="..."          # only needed if you swap to Google models

LANGFUSE_PUBLIC_KEY="..."     # from your Langfuse project settings
LANGFUSE_SECRET_KEY="..."
LANGFUSE_HOST="https://cloud.langfuse.com"

GROQ_API_KEY="..."            # from console.groq.com
```

---

## Course Lessons

### Lesson 1 — Base LLM Call

**File:** `01_base_llm_call.py`

The simplest possible interaction: send a system prompt and a user message to an LLM and print the response.

```
User ──► Groq API ──► Response ──► stdout
```

**Key concepts:**
- `Groq()` automatically reads `GROQ_API_KEY` from the environment.
- `chat.completions.create()` follows the OpenAI-compatible messages format: a list of `{"role": ..., "content": ...}` dicts.
- `temperature` controls randomness — lower values (e.g. `0.2`) make the model more deterministic; higher values (e.g. `0.9`) make it more creative.
- The actual text lives at `response.choices[0].message.content`.

**Run it:**
```bash
python 01_base_llm_call.py
```

---

### Lesson 2 — Logging with Langfuse

**File:** `02_base_call_plus_log.py`

Identical LLM call as Lesson 1, but now every call is automatically traced in Langfuse so you can inspect inputs, outputs, latency, and token usage from a web dashboard.

**Key concepts:**
- `@observe(name="...", as_type="generation")` is a decorator that wraps the function. Langfuse automatically captures the start time, end time, and return value of that function as a *trace* in its UI.
- `get_client().update_current_trace(metadata={...})` lets you attach arbitrary key-value metadata to the running trace (useful for tagging, filtering later).
- `langfuse.flush()` must be called at the end of a script (or process) to ensure all buffered traces are sent over the network before the process exits.

**Run it:**
```bash
python 02_base_call_plus_log.py
```

Then open your Langfuse dashboard → **Traces** to see the recorded call.

---

### Lesson 3 — Multi-step Agent

**File:** `03_multi_call.py`

A hand-rolled pipeline that breaks a task into smaller steps, executes each one in sequence (passing previous results as context), and synthesises a final answer — all fully traced in Langfuse.

```
Task ──► Plan ──► Execute Step 1 ──► Execute Step 2 ──► ... ──► Synthesise
             └─────────────── context flows forward ────────────────────────┘
```

**Key concepts:**
- **Nested `@observe` spans** — each sub-function (`_plan_steps`, `_execute_step`, `_synthesize_answer`) gets its own span inside the parent trace, giving you a flamegraph-style view.
- **`propagate_attributes(tags=[...])`** — attaches tags to all child spans created inside that `with` block without repeating yourself.
- **Context accumulation** — each `_execute_step` receives the outputs of all previous steps as a string, so later steps can build on earlier results.
- **Error handling** — the `except` block calls `update_current_span(level="ERROR", status_message=...)` so failures are visible and searchable in Langfuse.

**Run it:**
```bash
python 03_multi_call.py
```

---

### Lesson 4 — Datasets & Experiments

**File:** `04_dataset_experiment.py`

Introduces systematic evaluation: create a reusable benchmark dataset in Langfuse, run your LLM pipeline against every item, score the results automatically, and compare multiple model configurations side-by-side.

**Key concepts:**
- **Dataset** — a named collection of `(input, expected_output)` pairs stored in Langfuse. Create it once; reuse it across many experiments.
- **Experiment run** — one pass of your pipeline over the entire dataset. Each item's trace is linked to the run so you can filter by run name in the UI.
- **Evaluator** — a function that receives `(output, expected_output)` and returns one or more numeric scores (0.0 – 1.0). Scores are attached to the trace via `root_span.score_trace()`.
- **`get_client().run_experiment()`** — the high-level API that handles dataset iteration, linking, and score logging for you (recommended for production).
- **`compare_models()`** — runs the same dataset against different `(model, temperature)` configurations so you can compare them on the **Experiments** tab in Langfuse.

> **First run only:** Uncomment `create_sentiment_dataset()` in `__main__` to create the dataset. Comment it back out afterwards — calling it again will create a duplicate.

**Run it:**
```bash
python 04_dataset_experiment.py
```

---

### Lesson 5 — LLM as a Judge

**File:** `05_llm_as_a_judge.py`

Uses a *second* LLM call to evaluate the quality of the *first* LLM call. This replaces hard-coded rules with a flexible, nuanced judge that can reason about criteria like "is the reasoning well-justified?".

**Key concepts:**
- **Judge prompt engineering** — the judge's system prompt defines exact scoring criteria (correctness, reasoning quality, confidence calibration) and enforces JSON output for reliable parsing.
- **Low judge temperature (`0.1`)** — you want the judge to be consistent across runs, not creative.
- **Three separate scores** — splitting the evaluation into multiple dimensions (rather than one overall score) makes it easier to debug what exactly the model is getting wrong.
- **`Evaluation` objects** — Langfuse's typed wrapper for scores that includes an optional `comment` field for the judge's own explanation.
- **Prerequisite:** the `sentiment-benchmark-v1` dataset must exist (created in Lesson 4).

**Run it:**
```bash
python 05_llm_as_a_judge.py
```

---

### Lesson 6 — Tool Use with smolagents

**File:** `06_tool_use.py`

Gives the LLM access to real Python functions (tools) it can call at will. The agent decides *which* tools to call and *in what order* based on the user's request — you don't hard-code the logic.

**Architecture:**

```
User prompt
    │
    ▼
CodeAgent (LLM brain, smolagents)
    │  decides to call tools
    ├──► CalculateTool   ──► calculate()
    ├──► GetLogTool      ──► get_log()
    ├──► PolynomialRootsTool ──► get_polynomial_roots()
    ├──► WeatherTool     ──► get_weather()
    └──► StockTool       ──► get_stock_quantity()
    │
    ▼
Final answer (JSON)
    │
    ▼
Langfuse span (full trace including tool calls)
```

**Key concepts:**
- **`Tool` subclass** — each tool is a class with a `name`, `description`, typed `inputs` schema, and a `forward()` method. The description is what the LLM reads to decide whether to use the tool.
- **`CodeAgent`** — a smolagents agent that writes and executes small Python snippets to call tools, rather than just producing text. `max_steps` caps the number of tool-call rounds.
- **`LiteLLMModel`** — a thin wrapper that lets smolagents talk to any LiteLLM-compatible model; here it's pointed at Groq via the `groq/` prefix.
- **`langfuse.start_as_current_observation()`** — creates a parent span that wraps the entire agent run, so all tool calls appear as children in the Langfuse trace.
- **`@observe()` on tool functions** — each underlying Python function is also decorated, so individual tool executions appear as nested spans.

**Run it:**
```bash
python 06_tool_use.py
```

---

## How It All Fits Together

```
Lessons 1–2: The foundation
  └─ LLM call + observability

Lesson 3: Control flow
  └─ Chain multiple LLM calls with shared context

Lessons 4–5: Quality & evaluation
  └─ Measure how good your LLM pipeline actually is
     └─ With rules (Lesson 4) or another LLM (Lesson 5)

Lesson 6: Agency
  └─ Let the model choose its own actions via tools
```

All lessons share the same stack:

| Layer | Technology |
|---|---|
| LLM provider | Groq (fast inference, free tier) |
| Observability | Langfuse (traces, datasets, experiments) |
| Agent framework | smolagents (Lesson 6 only) |
| Config | `python-dotenv` + `.env` file |

---

## Tips & Best Practices

**General**

- **Always call `langfuse.flush()`** at the end of scripts or long-running jobs. Langfuse batches traces for efficiency; without `flush()` you may lose the last few traces when the process exits.
- **Use `.env` for secrets, never hard-code them.** The `.gitignore` already excludes `.env` — keep it that way.
- **Start with a low temperature for structured output** (JSON, step plans). Use higher temperatures only for creative generation.

**Prompting**

- Give the model a clear role in the system prompt ("You are an expert chef…") — it dramatically improves output quality.
- When you need JSON back, say so explicitly *and* give an example of the schema in the prompt.
- For multi-step pipelines, include the previous steps' results in the context so each step has the information it needs.

**Observability**

- Use `as_type="generation"` on functions that make LLM calls, and plain `@observe()` on orchestration functions. This gives Langfuse the right metadata to display token counts and costs.
- Add meaningful `metadata` to traces (`update_current_trace`) — model name, user ID, feature flag, etc. These become filterable fields in the Langfuse UI.
- Tag experiments consistently (e.g. `experiment_name = f"{model}-t{temperature}"`) so the Experiments comparison view is readable.

**Evaluation**

- **Create the dataset once, then reuse it.** Datasets are versioned in Langfuse; running the creation function again creates a duplicate.
- **Separate your evaluator from your task function.** This makes it easy to swap evaluators (rule-based → LLM judge) without touching the pipeline.
- **Use a different (or at least identically-prompted) model as judge.** If you use the same model with the same prompt to both produce and judge output, you introduce bias.
- **Log multiple score dimensions** instead of a single overall score. It makes debugging much faster.

**Tool use**

- **Write descriptive tool descriptions.** The LLM reads them verbatim to decide whether to call a tool. Vague descriptions lead to wrong or missed tool calls.
- **Cap `max_steps`** to avoid runaway loops — `10` is a safe default for most tasks.
- **Wrap tool internals in `@observe()`** so their execution appears in the Langfuse trace alongside the agent's reasoning steps.
- **Test tools independently** (call the plain Python function first) before plugging them into the agent. Debugging inside an agent loop is harder.

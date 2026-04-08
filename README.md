# 📧 AI Email Intelligence Environment

> An OpenEnv-compliant multi-task AI environment for email classification, summarization, and reply generation.

---

## 🧠 Motivation — The Email Overload Problem

The average professional receives **120+ emails per day**. Studies show that knowledge workers spend up to **28% of their workweek** just reading and responding to email. This cognitive load leads to:

- Important messages getting buried under spam
- Delayed responses to time-sensitive requests
- Poorly written replies that damage professional relationships
- Decision fatigue from triaging low-value messages

The **AI Email Intelligence Environment** addresses this by providing a structured benchmark where AI agents can be evaluated on three core email management skills — classification, summarization, and reply generation — in a reproducible, reward-driven framework.

---

## 🗂️ Project Structure

```
email_env/
├── __init__.py              # Package entry point
├── env.py                   # OpenEnv-compliant environment core
├── models.py                # Pydantic models: Observation, Action, Reward
├── main.py                  # Interactive demo / CLI
├── inference.py             # Baseline LLM inference script
├── openenv.yaml             # OpenEnv compliance manifest
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build definition
├── setup.py                 # Package installation config
├── tasks/
│   ├── __init__.py
│   ├── classification.py    # 10 labelled email samples
│   ├── summarization.py     # 5 long-email + reference pairs
│   └── reply.py             # 5 customer email + rubric pairs
├── graders/
│   ├── __init__.py
│   ├── classification_grader.py   # Exact-match ± penalty
│   ├── summarization_grader.py    # Keyword + F1 partial scoring
│   └── reply_grader.py            # Relevance + tone + format scoring
└── tests/
    ├── __init__.py
    └── test_env.py          # 25+ unit tests
```

---

## 🎯 Task Descriptions

### Task 1 — Email Classification (Easy)
| Property | Detail |
|----------|--------|
| **Input** | Raw email text |
| **Output** | `spam` \| `important` \| `normal` |
| **Grader** | Deterministic exact-match |
| **Reward** | +1.0 correct, −0.2 wrong label, −0.5 invalid |
| **Samples** | 10 labelled emails |

The agent must correctly classify emails into one of three categories. This tests basic language understanding and spam/priority detection.

**Example:**
```
Email: "Congratulations! You've won a FREE iPhone..."
Expected: spam
```

---

### Task 2 — Email Summarization (Medium)
| Property | Detail |
|----------|--------|
| **Input** | Multi-paragraph email (150–300 words) |
| **Output** | 1–2 sentence summary |
| **Grader** | Keyword coverage + unigram F1 similarity |
| **Reward** | Partial scoring 0.0–1.0 |
| **Samples** | 5 emails with reference summaries |

The agent must distill a long email into a concise, accurate summary. Scored along three weighted axes: keyword coverage (50%), semantic similarity (30%), and length appropriateness (20%).

**Example:**
```
Email: "...product launch scheduled for March 15th... QA testing by March 8th..."
Good Summary: "VP of Product reminds teams of March 15 launch deadlines including QA by March 8."
Score: ~0.85
```

---

### Task 3 — Email Reply Generation (Hard)
| Property | Detail |
|----------|--------|
| **Input** | Customer email with complaints / requests |
| **Output** | Full professional reply email |
| **Grader** | Relevance (35%) + Tone (30%) + Completeness (20%) + Format (15%) |
| **Reward** | Partial scoring 0.0–1.0 |
| **Samples** | 5 customer emails with rubrics |

The agent must write a professional, empathetic reply that addresses all customer concerns. Scored on required content coverage, professional tone (with forbidden-phrase detection), semantic similarity to a reference reply, and correct email format.

**Example:**
```
Customer: "I received the wrong item and need this fixed NOW!"
Good Reply: "Dear [Name], I sincerely apologize for receiving the wrong item. We will
             immediately ship your correct order via overnight delivery at no cost..."
```

---

## 🔄 Observation & Action Space

### Observation Format
```json
{
  "task_type": "classification | summarization | reply",
  "email_text": "Full email text...",
  "task_id": "cls_001",
  "instructions": "Human-readable task instructions"
}
```

### Action Format
```json
{
  "response": "spam"
}
```
For classification, this is a single word. For summarization, 1–2 sentences. For reply, a full email.

### Reward Format
```json
{
  "score": 0.85,
  "feedback": "Keyword coverage: 75%... Semantic similarity: 82%... Final score: 0.8500",
  "partial": true
}
```

---

## 🔌 OpenEnv API

```python
from email_env import EmailEnvironment, Action

env = EmailEnvironment(seed=42)

# Reset — returns first Observation
obs = env.reset()

# Step — returns (Observation, Reward, done, info)
action = Action(response="spam")
next_obs, reward, done, info = env.step(action)

print(reward.score)     # e.g. 1.0
print(reward.feedback)  # "Correct! The email is 'spam'."

# State snapshot
state = env.state()
print(state.cumulative_score)
print(state.current_task_index)
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### 1. Clone & Install
```bash
git clone https://github.com/your-org/ai-email-intelligence-environment
cd ai-email-intelligence-environment

# Install the package
pip install -e .

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run the Demo (no API key needed)
```bash
python email_env/main.py
```

This runs a full episode with pre-written demo answers and prints step-by-step scores.

```bash
# Run with random (wrong) answers to see penalty behavior
python email_env/main.py --random
```

### 3. Run Tests
```bash
pytest email_env/tests/ -v
```

### 4. Run Baseline Inference (requires API key)
```bash
# Using OpenAI
export HF_TOKEN="sk-your-openai-key"
python email_env/inference.py --provider openai

# Using Hugging Face Inference API
export HF_TOKEN="hf_your_token"
python email_env/inference.py --provider hf

# Save results to JSON
python email_env/inference.py --provider openai --output results.json
```

---

## 🐳 Docker

### Build the Image
```bash
docker build -t ai-email-env .
```

### Run the Demo (no API key)
```bash
docker run --rm ai-email-env python email_env/main.py
```

### Run Baseline Inference
```bash
docker run --rm \
  -e HF_TOKEN=your_api_key_here \
  ai-email-env \
  python email_env/inference.py --provider openai
```

### Run Tests
```bash
docker run --rm ai-email-env pytest email_env/tests/ -v
```

### Interactive Shell
```bash
docker run --rm -it ai-email-env bash
```

---

## 📊 Baseline Results

Results using `gpt-4o-mini` (OpenAI) as the baseline agent:

| Task | Samples | Avg Score | Notes |
|------|---------|-----------|-------|
| Classification | 10 | ~0.72 | Struggles with edge-case phrasing |
| Summarization  | 5  | ~0.55 | Misses specific dates/numbers |
| Reply          | 5  | ~0.48 | Tone good; misses some required elements |
| **Overall**    | **20** | **~0.62** | |

> Scores are approximate and will vary with model version and temperature.

---

## 🏗️ Reward Design Rationale

| Principle | Implementation |
|-----------|----------------|
| **Partial credit** | Summarization and reply graders award 0–1 based on weighted sub-scores |
| **Penalise wrong answers** | Classification: −0.2 for wrong label, −0.5 for invalid output |
| **Penalise bad tone** | Reply grader deducts 0.3 per forbidden phrase |
| **Penalise loops** | −0.1 if agent submits identical response for the same task twice |
| **Reproducibility** | All graders are deterministic; `seed=42` ensures fixed task order |

---

## 🧩 Extending the Environment

### Adding New Email Samples
```python
# In tasks/classification.py
CLASSIFICATION_SAMPLES.append(ClassificationSample(
    task_id="cls_011",
    email_text="Your invoice #1234 is due on Friday...",
    label="important",
))
```

### Custom Grading Logic
Each grader is a pure function — swap it out by modifying `env.py`:
```python
from my_custom_grader import grade_classification
reward = grade_classification(response, task["label"])
```

---

## 📄 License

MIT License — see LICENSE for details.

---

## 🏆 Hackathon Compliance Checklist

- [x] OpenEnv `step()`, `reset()`, `state()` API
- [x] Pydantic models for Observation, Action, Reward
- [x] `openenv.yaml` manifest
- [x] 3 tasks (easy / medium / hard)
- [x] Deterministic graders (0.0–1.0 range)
- [x] Partial scoring
- [x] Loop penalty
- [x] Baseline inference script using `HF_TOKEN`
- [x] Dockerfile that builds and runs
- [x] `requirements.txt`
- [x] Full test suite
- [x] This README

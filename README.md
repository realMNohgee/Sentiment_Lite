# 💬 Sentiment_Lite — Zero-Dependency Sentiment Scoring
![CI](https://github.com/realMNohgee/Sentiment_Lite/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

**"How does this text feel?"** A single-file CLI that scores the sentiment of any text — a sentence, a review, a support ticket, a tweet — using an embedded ~120 positive / ~120 negative English lexicon with 1–3 intensity weights and simple negation handling. Pure Python standard library. No pip installs, no third-party imports.

🧰 **[Tool on Hermtica Marketplace](https://hermtica.com/marketplace)**

## One tool, many domains

| Domain | What Sentiment_Lite does for you |
|---|---|
| 🛍️ **Product & reviews** | Triage thousands of reviews into positive/negative/neutral buckets before routing to humans |
| 🎧 **Customer support** | Flag angry tickets for escalation and happy ones for CSAT surveys |
| 🐦 **Social listening** | Score posts, comments, and mentions for brand-health pulse checks |
| 🧠 **Agentic AI** | Give an agent a fast, dependency-free sentiment gate to decide tone before it replies |
| 📈 **Market research** | Trend sentiment of survey free-text responses without an ML pipeline |
| 📝 **Content QA** | Check that generated copy lands on the intended emotional register |

## Agentic AI framing

Autonomous agents act on text they don't have time to *feel*. **Sentiment_Lite** hands an agent a deterministic, offline `score`/`tokens` primitive it can drop into any loop: `--format json` for structured consumption, nonzero exit codes for CI gates, zero deps so it runs anywhere Python 3.9+ runs. An agent can sort retrieved reviews by polarity, decide whether a support reply needs empathy, or gate its own generated output — all from one stdlib script, no token budget spent on an LLM call for a job a lexicon already does.

## Install

```bash
git clone git@github.com:realMNohgee/Sentiment_Lite.git
cd Sentiment_Lite
# That's it. Zero dependencies. Pure Python stdlib.
```

## Quick start

```bash
python3 Sentiment_Lite.py score "I love this excellent product"
python3 Sentiment_Lite.py score "This is not good at all"
python3 Sentiment_Lite.py score reviews.txt            # score a file
cat tweet.txt | python3 Sentiment_Lite.py score -      # score stdin
python3 Sentiment_Lite.py tokens "not bad, not terrible"
```

## Usage

```
python3 Sentiment_Lite.py score  INPUT [--format text|json]
python3 Sentiment_Lite.py tokens INPUT [--format text|json]
```

- `INPUT` is an **existing file path**, **quoted text**, or **`-`** for stdin.
- `--format` works **before or after** the subcommand.
- Missing/empty input (empty text, empty file, empty stdin) exits **nonzero**.

### `score` — polarity, label, confidence

```bash
$ python3 Sentiment_Lite.py score "I love this excellent product"
label: positive
polarity: 0.857
confidence: 0.857
positive: 6
negative: 0
positive_words: 2
negative_words: 0
total_words: 6
```

Polarity is `(pos − neg) / (pos + neg + 1)`, bounded in (−1, +1). Confidence is
evidence-based: `1 − 1/(1 + pos + neg)`, so it starts at 0 with no sentiment
words and climbs toward 1 as weighted evidence accumulates.

```bash
$ python3 Sentiment_Lite.py --format json score "not good"
{
  "label": "negative",
  "polarity": -0.667,
  "confidence": 0.667,
  "positive": 0,
  "negative": 2,
  "positive_words": 0,
  "negative_words": 1,
  "total_words": 3
}
```

### `tokens` — per-word breakdown

```bash
$ python3 Sentiment_Lite.py tokens "not good, but not terrible"
word      lexicon   weight  contribution  negated
good      positive  2       -2            yes
terrible  negative  3       +3            yes
```

Negations (`not`, `no`, `never`, `without`, and contractions like `don't`,
`isn't`) flip the sign of the next sentiment word: `not bad` scores positive.

## How it works

1. **Tokenize** — lowercase and split into word tokens (letters + apostrophes).
2. **Negation** — a negation word arms a flip that applies to the *next*
   sentiment word, so `not very good` and `don't feel great` negate correctly.
3. **Score** — sum positive and negative lexicon hits (each weighted 1–3 by
   intensity), then compute polarity and classify into negative / neutral /
   positive (a ±0.1 neutral band) with an evidence-based confidence.

Lexicon-based scoring is intentionally lightweight: it is fast, transparent,
and dependency-free — not a substitute for a trained model on nuanced or
sarcastic text, but a solid, inspectable first signal.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (including a valid neutral result) |
| 1 | Missing/empty input, or unreadable/missing file |
| 2 | Bad usage (e.g. missing INPUT, invalid `--format`) |

## License

MIT — see [LICENSE](LICENSE).

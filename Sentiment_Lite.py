from __future__ import annotations

# Sentiment_Lite — zero-dependency sentiment scoring for text.
# Pure Python standard library only. No third-party dependencies.
#
# Subcommands:
#   score   INPUT    -> polarity score, label, and confidence for text or a file
#   tokens  INPUT    -> per-token breakdown of every sentiment-bearing word
#
# INPUT is a file path (existing) or quoted text. "-" reads from stdin.

import argparse
import json
import os
import re
import sys

VERSION = "1.0.0"

# Explicit description constant (NOT __doc__): the `from __future__ import
# annotations` line above must be the first line of code, which pushes any
# module docstring below it and leaves __doc__ empty in some setups.
_DESCRIPTION = (
    "Sentiment_Lite — zero-dependency sentiment scoring for text.\n"
    "Scores polarity with an embedded positive/negative English lexicon and\n"
    "simple negation handling (not/no/never/without flip the next sentiment\n"
    "word's sign). Pure Python standard library."
)

# ---------------------------------------------------------------------------
# Embedded sentiment lexicon
# ---------------------------------------------------------------------------
# Each entry maps a lowercase word to an intensity weight (1 = mild,
# 2 = moderate, 3 = strong). Roughly 120 positive and 120 negative words,
# hand-picked for breadth across everyday, product-review, and social text.
_POSITIVE = {
    # intensity 3 — strongly positive
    "love": 3, "adore": 3, "excellent": 3, "amazing": 3, "fantastic": 3,
    "wonderful": 3, "brilliant": 3, "outstanding": 3, "superb": 3,
    "exceptional": 3, "incredible": 3, "phenomenal": 3, "magnificent": 3,
    "perfect": 3, "glorious": 3, "spectacular": 3, "awesome": 3,
    "delightful": 3, "marvelous": 3, "terrific": 3, "splendid": 3,
    "extraordinary": 3, "fabulous": 3, "tremendous": 3, "exquisite": 3,
    "heavenly": 3, "blissful": 3, "triumphant": 3, "masterful": 3,
    "stellar": 3, "unbeatable": 3, "breathtaking": 3, "radiant": 3,
    "euphoric": 3, "flawless": 3,
    # intensity 2 — moderately positive
    "happy": 2, "great": 2, "good": 2, "nice": 2, "beautiful": 2,
    "joy": 2, "joyful": 2, "pleased": 2, "satisfied": 2, "enjoy": 2,
    "grateful": 2, "thankful": 2, "cheerful": 2, "optimistic": 2,
    "hopeful": 2, "excited": 2, "pleasant": 2, "lovely": 2, "proud": 2,
    "confident": 2, "calm": 2, "peaceful": 2, "kind": 2, "generous": 2,
    "friendly": 2, "sweet": 2, "warm": 2, "encouraging": 2, "supportive": 2,
    "positive": 2, "delighted": 2, "thrilled": 2, "impressed": 2,
    "admirable": 2, "charming": 2, "elegant": 2, "graceful": 2,
    "vibrant": 2, "refreshing": 2, "soothing": 2, "comforting": 2,
    "uplifting": 2, "inspiring": 2, "motivating": 2, "heartwarming": 2,
    "enchanting": 2, "appealing": 2, "attractive": 2, "favorable": 2,
    "blessed": 2, "fortunate": 2, "lucky": 2, "prosperous": 2,
    "thriving": 2, "flourishing": 2, "successful": 2, "accomplished": 2,
    "skillful": 2, "talented": 2, "gifted": 2, "like": 2, "admire": 2,
    "appreciate": 2,
    # intensity 1 — mildly positive
    "okay": 1, "fine": 1, "decent": 1, "agreeable": 1, "acceptable": 1,
    "satisfactory": 1, "alright": 1, "glad": 1, "welcome": 1,
    "appreciated": 1, "promising": 1, "reassuring": 1, "fair": 1,
    "tidy": 1, "neat": 1, "useful": 1, "handy": 1, "convenient": 1,
    "solid": 1, "steady": 1, "stable": 1, "balanced": 1, "healthy": 1,
    "safe": 1, "content": 1,
}

_NEGATIVE = {
    # intensity 3 — strongly negative
    "hate": 3, "loathe": 3, "despise": 3, "detest": 3, "terrible": 3,
    "awful": 3, "horrible": 3, "atrocious": 3, "dreadful": 3,
    "disgusting": 3, "revolting": 3, "repulsive": 3, "abominable": 3,
    "horrific": 3, "horrendous": 3, "appalling": 3, "devastating": 3,
    "catastrophic": 3, "disastrous": 3, "tragic": 3, "heartbreaking": 3,
    "miserable": 3, "wretched": 3, "deplorable": 3, "shameful": 3,
    "outrageous": 3, "heinous": 3, "vile": 3, "sinister": 3,
    "malicious": 3, "evil": 3, "cruel": 3, "brutal": 3, "savage": 3,
    "ruthless": 3, "merciless": 3, "toxic": 3, "poisonous": 3,
    "ruinous": 3, "destructive": 3, "fatal": 3,
    # intensity 2 — moderately negative
    "bad": 2, "poor": 2, "sad": 2, "unhappy": 2, "angry": 2,
    "upset": 2, "depressed": 2, "disappointed": 2, "frustrated": 2,
    "annoyed": 2, "irritated": 2, "worried": 2, "anxious": 2,
    "stressed": 2, "afraid": 2, "scared": 2, "fearful": 2, "gloomy": 2,
    "grim": 2, "bleak": 2, "hopeless": 2, "desperate": 2, "pathetic": 2,
    "lousy": 2, "inferior": 2, "flawed": 2, "defective": 2, "broken": 2,
    "damaged": 2, "harmful": 2, "hurtful": 2, "painful": 2, "nasty": 2,
    "mean": 2, "rude": 2, "hostile": 2, "aggressive": 2, "offensive": 2,
    "unpleasant": 2, "ugly": 2, "boring": 2, "dull": 2, "tedious": 2,
    "exhausting": 2, "annoying": 2, "irritating": 2, "bothersome": 2,
    "troublesome": 2, "problematic": 2, "confusing": 2, "messy": 2,
    "chaotic": 2, "weak": 2, "wrong": 2, "unfair": 2, "dislike": 2,
    # intensity 1 — mildly negative
    "mediocre": 1, "subpar": 1, "inadequate": 1, "insufficient": 1,
    "lacking": 1, "shoddy": 1, "unfavorable": 1, "negative": 1,
    "discouraged": 1, "uneasy": 1, "tense": 1, "nervous": 1,
    "skeptical": 1, "doubtful": 1, "dubious": 1, "suspicious": 1,
    "regretful": 1, "sorry": 1, "guilty": 1, "ashamed": 1,
    "embarrassed": 1, "neglectful": 1, "careless": 1, "weary": 1,
    "dismal": 1,
}

# Words (and contractions) that flip the sign of the NEXT sentiment word.
# Pending negation persists across neutral words until a sentiment word
# consumes it, so "not very good" and "don't feel great" both flip correctly.
_NEGATIONS = {
    "not", "no", "never", "without", "nor", "neither", "cannot", "cant",
    "n't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "ain't",
}

# File extensions that signal "this INPUT was meant to be a path", so a
# non-existent .txt/.md/etc. argument is reported as a missing file rather
# than silently scored as literal text.
_PATH_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".log",
    ".html", ".text", ".yaml", ".yml",
)

# Polarity magnitude beyond which the label leaves "neutral".
_LABEL_THRESHOLD = 0.1

# Tokenizer: lowercase and keep runs of ASCII letters and apostrophes so that
# contractions like "don't" survive as a single token for negation handling.
_WORD_RE = re.compile(r"[a-zA-Z']+")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list:
    """Split text into lowercase word tokens (letters + apostrophes only)."""
    return _WORD_RE.findall(text.lower())


def _classify(polarity: float) -> str:
    """Map a polarity score to a human label using a small neutral band."""
    if polarity > _LABEL_THRESHOLD:
        return "positive"
    if polarity < -_LABEL_THRESHOLD:
        return "negative"
    return "neutral"


def _analyze(text: str) -> dict:
    """Score the sentiment of `text` and collect per-token evidence.

    Returns a dict with the polarity score, label, evidence-based confidence,
    weighted positive/negative totals, hit counts, and an ordered list of
    sentiment-bearing tokens (including their signed contribution after any
    negation flip).
    """
    words = _tokenize(text)
    pos = 0            # total positive weight (negated negative words add here)
    neg = 0            # total negative weight (negated positive words add here)
    pos_hits = 0       # count of positive-lexicon hits
    neg_hits = 0       # count of negative-lexicon hits
    negate_next = False  # a pending negation awaiting the next sentiment word
    hits = []          # ordered per-token evidence

    for word in words:
        # A negation word arms the flip; it never contributes sentiment itself.
        if word in _NEGATIONS:
            negate_next = True
            continue

        if word in _POSITIVE:
            weight = _POSITIVE[word]
            hit = {"word": word, "lexicon": "positive", "weight": weight,
                   "contribution": weight, "negated": False}
            if negate_next:
                neg += weight      # flipped: positive word now counts negative
                neg_hits += 1
                hit["contribution"] = -weight
                hit["negated"] = True
            else:
                pos += weight
                pos_hits += 1
            negate_next = False    # negation consumed by this sentiment word
            hits.append(hit)
            continue

        if word in _NEGATIVE:
            weight = _NEGATIVE[word]
            hit = {"word": word, "lexicon": "negative", "weight": weight,
                   "contribution": -weight, "negated": False}
            if negate_next:
                pos += weight      # flipped: negative word now counts positive
                pos_hits += 1
                hit["contribution"] = weight
                hit["negated"] = True
            else:
                neg += weight
                neg_hits += 1
            negate_next = False
            hits.append(hit)
            continue

        # Neutral word: keep negate_next so a negation still reaches the next
        # sentiment word (e.g. "not very good" -> "good" gets flipped).

    # Polarity in (-1, 1): +1 smoothing keeps the denominator nonzero and the
    # score bounded even for a single word.
    polarity = (pos - neg) / float(pos + neg + 1)
    # Confidence grows with total weighted evidence (0 = no sentiment words).
    confidence = 1.0 - 1.0 / float(1 + pos + neg)

    return {
        "label": _classify(polarity),
        "polarity": round(polarity, 3),
        "confidence": round(confidence, 3),
        "positive": pos,
        "negative": neg,
        "positive_words": pos_hits,
        "negative_words": neg_hits,
        "total_words": len(words),
        "tokens": hits,
    }


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _looks_like_path(raw: str) -> bool:
    """Heuristic: does `raw` read like a file path rather than free text?"""
    if raw.startswith("~") or raw.startswith("."):
        return True
    if "/" in raw or "\\" in raw:
        return True
    return raw.lower().endswith(_PATH_EXTENSIONS)


def _resolve_input(raw: str):
    """Turn the INPUT argument into text, or return None when input is missing.

    Resolution order: stdin marker "-", existing file, directory (error),
    nonexistent path-shaped argument (error), otherwise the literal text.
    Returns None (caller errors out) for empty text / empty file / empty stdin.
    """
    if raw == "-":
        data = sys.stdin.read()
        return data if data.strip() else None  # empty stdin -> missing input

    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as fh:
            data = fh.read()
        return data if data.strip() else None  # empty file -> missing input

    if os.path.isdir(raw):
        raise OSError(f"'{raw}' is a directory")

    if _looks_like_path(raw):
        raise OSError(f"file not found: '{raw}'")

    # Not an existing path and doesn't look like one -> treat as literal text.
    return raw if raw.strip() else None


def _read_input(args) -> str:
    """Resolve INPUT and return the text, or None after printing an error.

    Two distinct failure messages are handled here so callers never double-
    report: a file/path error (OSError) prints its own message, while genuinely
    empty/missing input prints the "no input" message. Returns None in both
    cases so the handler can exit nonzero.
    """
    try:
        text = _resolve_input(args.input)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    if text is None:
        print("Error: no input provided (empty text, empty file, or empty "
              "stdin).", file=sys.stderr)
        return None
    return text


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _emit_score(result: dict, fmt: str) -> None:
    """Print the score summary in text or JSON."""
    if fmt == "json":
        # Omit the verbose per-token list; `tokens` subcommand exposes that.
        obj = {k: result[k] for k in (
            "label", "polarity", "confidence", "positive", "negative",
            "positive_words", "negative_words", "total_words")}
        print(json.dumps(obj, indent=2))
        return

    print("label: %s" % result["label"])
    print("polarity: %.3f" % result["polarity"])
    print("confidence: %.3f" % result["confidence"])
    print("positive: %d" % result["positive"])
    print("negative: %d" % result["negative"])
    print("positive_words: %d" % result["positive_words"])
    print("negative_words: %d" % result["negative_words"])
    print("total_words: %d" % result["total_words"])


def _emit_tokens(result: dict, fmt: str) -> None:
    """Print the per-token sentiment breakdown in text or JSON."""
    tokens = result["tokens"]

    if fmt == "json":
        print(json.dumps(tokens, indent=2))
        return

    if not tokens:
        print("No sentiment tokens found.")
        return

    # Build an aligned table with a header row.
    rows = [("word", "lexicon", "weight", "contribution", "negated")]
    for t in tokens:
        contrib = "%+d" % t["contribution"]
        rows.append((t["word"], t["lexicon"], str(t["weight"]), contrib,
                     "yes" if t["negated"] else "no"))

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    """`score INPUT`: print polarity, label, and confidence."""
    text = _read_input(args)
    if text is None:
        return 1
    _emit_score(_analyze(text), args.format)
    return 0


def cmd_tokens(args: argparse.Namespace) -> int:
    """`tokens INPUT`: list each sentiment-bearing token and its contribution."""
    text = _read_input(args)
    if text is None:
        return 1
    _emit_tokens(_analyze(text), args.format)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the parser with `--format` accepted before OR after the subcommand.

    The shared flag lives on a parent parser with `default=argparse.SUPPRESS`
    and is attached to BOTH the top-level parser and every subparser. SUPPRESS
    means "leave the attribute unset unless the flag was actually given", so a
    subparser that never sees `--format` doesn't reset a value supplied before
    the subcommand. Fallbacks are resolved once in main().
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", choices=["text", "json"],
                        default=argparse.SUPPRESS,
                        help="Output format (default: text).")

    parser = argparse.ArgumentParser(
        prog="Sentiment_Lite",
        description=_DESCRIPTION,
        parents=[common],                    # so --format works before subcommand
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # score INPUT
    sp = sub.add_parser("score", parents=[common],
                        help="Score the sentiment of text or a file.")
    sp.add_argument("input", help="File path, quoted text, or '-' for stdin.")
    sp.set_defaults(func=cmd_score)

    # tokens INPUT
    tp = sub.add_parser("tokens", parents=[common],
                        help="List each sentiment-bearing token.")
    tp.add_argument("input", help="File path, quoted text, or '-' for stdin.")
    tp.set_defaults(func=cmd_tokens)

    return parser


def main(argv=None) -> int:
    """Parse args once, resolve the shared-flag fallback, then dispatch."""
    args = build_parser().parse_args(argv)

    # SUPPRESS keeps --format unset unless explicitly provided; resolve it.
    args.format = getattr(args, "format", None) or "text"

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

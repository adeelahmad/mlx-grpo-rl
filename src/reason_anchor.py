"""
ReasonAnchor v2.0 — Heuristic Reasoning Anchor Generator for RL Fine-Tuning
=============================================================================

Generates a single, compact reasoning scaffold string from (prompt, target_cot, answer)
using Self-Discover-inspired module selection + adaptive CoT compression.

Zero dependencies beyond Python 3.9+ stdlib. No model needed. Pure heuristic.

Architecture:
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ DETECT  │ → │ EXTRACT │ → │ SELECT  │ → │ ADAPT   │ → │ COMPOSE │
  │ problem │   │ moves   │   │ modules │   │ to task │   │ anchor  │
  │ types   │   │ from CoT│   │ (SelfD) │   │         │   │ string  │
  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘

Usage:
  # Single example
  result = generate_anchor(prompt, cot, answer)
  print(result.anchor)

  # Batch from JSONL
  python reason_anchor.py --input data.jsonl --output anchors.jsonl

  # Demo
  python reason_anchor.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROBLEM TYPE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


class ProblemType(Enum):
    MATH = auto()
    CODE = auto()
    LOGIC = auto()
    SCIENCE = auto()
    MEDICAL = auto()
    LEGAL = auto()
    FINANCIAL = auto()
    BUSINESS = auto()
    CREATIVE = auto()
    FACTUAL = auto()
    REASONING = auto()
    MULTI_STEP = auto()


class ReasoningMove(Enum):
    """Atomic reasoning operations detected in CoT."""

    DECOMPOSE = "decompose"
    CALCULATE = "calculate"
    COMPARE = "compare"
    INFER = "infer"
    VERIFY = "verify"
    HYPOTHESIZE = "hypothesize"
    ELIMINATE = "eliminate"
    ANALOGIZE = "analogize"
    REFRAME = "reframe"
    BACKTRACK = "backtrack"
    SYNTHESIZE = "synthesize"
    CONSTRAIN = "constrain"
    GENERALIZE = "generalize"
    SPECIALIZE = "specialize"
    TRACE = "trace"  # code tracing


class AnchorStyle(Enum):
    TELEGRAPHIC = "telegraphic"  # ultra-compact
    STRUCTURED = "structured"  # labeled steps
    NARRATIVE = "narrative"  # natural flow
    SCAFFOLDED = "scaffolded"  # meta-reasoning + steps


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REASONING MODULE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReasoningModule:
    id: str
    name: str
    trigger_keywords: list[str]
    trigger_types: list[ProblemType]
    trigger_moves: list[ReasoningMove]
    template: str
    weight: float = 1.0


MODULES: list[ReasoningModule] = [
    ReasoningModule(
        id="DECOMP",
        name="Decompose",
        trigger_keywords=["complex", "multiple", "parts", "steps", "components", "break down"],
        trigger_types=[ProblemType.MULTI_STEP, ProblemType.CODE, ProblemType.MATH],
        trigger_moves=[ReasoningMove.DECOMPOSE],
        template="Break into sub-problems → solve each → combine.",
        weight=1.5,
    ),
    ReasoningModule(
        id="FP",
        name="First Principles",
        trigger_keywords=[
            "why",
            "fundamental",
            "basic",
            "root",
            "underlying",
            "assume",
            "how does",
            "explain",
        ],
        trigger_types=[ProblemType.REASONING, ProblemType.SCIENCE, ProblemType.MATH],
        trigger_moves=[ReasoningMove.DECOMPOSE, ReasoningMove.INFER],
        template="Start from known facts → build up reasoning chain.",
        weight=1.4,
    ),
    ReasoningModule(
        id="CALC",
        name="Step Calculation",
        trigger_keywords=[
            "calculate",
            "compute",
            "how many",
            "how much",
            "total",
            "sum",
            "product",
            "difference",
            "ratio",
            "percent",
            "average",
            "equation",
            "solve",
            "find the value",
        ],
        trigger_types=[ProblemType.MATH],
        trigger_moves=[ReasoningMove.CALCULATE],
        template="Given → operation → intermediate → operation → result.",
        weight=1.6,
    ),
    ReasoningModule(
        id="ELIM",
        name="Elimination",
        trigger_keywords=[
            "which",
            "choose",
            "option",
            "select",
            "possible",
            "candidate",
            "eliminate",
            "rule out",
            "narrow",
            "who has",
        ],
        trigger_types=[ProblemType.LOGIC, ProblemType.REASONING],
        trigger_moves=[ReasoningMove.ELIMINATE, ReasoningMove.COMPARE],
        template="List options → eliminate invalid → remaining = answer.",
        weight=1.3,
    ),
    ReasoningModule(
        id="HYPO",
        name="Hypothesis Test",
        trigger_keywords=[
            "if",
            "suppose",
            "assume",
            "hypothesis",
            "test",
            "verify",
            "check",
            "confirm",
            "valid",
            "try",
        ],
        trigger_types=[ProblemType.SCIENCE, ProblemType.LOGIC, ProblemType.REASONING],
        trigger_moves=[ReasoningMove.HYPOTHESIZE, ReasoningMove.VERIFY],
        template="Hypothesize → test → outcome → conclude.",
        weight=1.2,
    ),
    ReasoningModule(
        id="CSP",
        name="Constraint Satisfaction",
        trigger_keywords=[
            "constraint",
            "condition",
            "must",
            "requirement",
            "satisfy",
            "given that",
            "subject to",
            "rule",
            "doesn't",
            "can't",
            "different",
            "each",
        ],
        trigger_types=[ProblemType.LOGIC, ProblemType.MATH, ProblemType.CODE],
        trigger_moves=[ReasoningMove.CONSTRAIN, ReasoningMove.ELIMINATE],
        template="List constraints → apply sequentially → find valid assignment.",
        weight=1.4,
    ),
    ReasoningModule(
        id="PATTERN",
        name="Pattern Recognition",
        trigger_keywords=[
            "pattern",
            "sequence",
            "series",
            "next",
            "trend",
            "recurring",
            "follows",
            "observe",
            "notice",
        ],
        trigger_types=[ProblemType.MATH, ProblemType.REASONING],
        trigger_moves=[ReasoningMove.GENERALIZE, ReasoningMove.INFER],
        template="Observe examples → identify pattern → apply.",
        weight=1.2,
    ),
    ReasoningModule(
        id="ANALOG",
        name="Analogy",
        trigger_keywords=["like", "similar", "analogy", "compare", "just as", "resembles"],
        trigger_types=[ProblemType.REASONING, ProblemType.CREATIVE, ProblemType.SCIENCE],
        trigger_moves=[ReasoningMove.ANALOGIZE],
        template="Map to known domain → transfer insight → conclude.",
        weight=1.0,
    ),
    ReasoningModule(
        id="VERIFY",
        name="Verification",
        trigger_keywords=[
            "check",
            "verify",
            "confirm",
            "make sure",
            "double-check",
            "sanity",
            "validate",
        ],
        trigger_types=list(ProblemType),
        trigger_moves=[ReasoningMove.VERIFY],
        template="Check: method → expected vs actual → confirm.",
        weight=1.1,
    ),
    ReasoningModule(
        id="BACK",
        name="Backtrack",
        trigger_keywords=[
            "wait",
            "actually",
            "mistake",
            "wrong",
            "reconsider",
            "hmm",
            "correction",
        ],
        trigger_types=list(ProblemType),
        trigger_moves=[ReasoningMove.BACKTRACK],
        template="Spot error → correct → revise conclusion.",
        weight=1.3,
    ),
    ReasoningModule(
        id="SYNTH",
        name="Synthesize",
        trigger_keywords=[
            "therefore",
            "combining",
            "together",
            "overall",
            "conclusion",
            "putting it all",
            "synthesis",
            "final",
        ],
        trigger_types=list(ProblemType),
        trigger_moves=[ReasoningMove.SYNTHESIZE],
        template="Combine sub-results → integrate → final answer.",
        weight=1.0,
    ),
    ReasoningModule(
        id="TRACE",
        name="Code Trace",
        trigger_keywords=[
            "output",
            "program",
            "execute",
            "trace",
            "variable",
            "loop",
            "return",
            "algorithm",
            "debug",
            "print",
            "function call",
        ],
        trigger_types=[ProblemType.CODE],
        trigger_moves=[ReasoningMove.TRACE, ReasoningMove.CALCULATE],
        template="Trace execution line by line → track state → determine output.",
        weight=1.5,
    ),
    ReasoningModule(
        id="CMP",
        name="Compare",
        trigger_keywords=[
            "better",
            "worse",
            "advantage",
            "disadvantage",
            "trade-off",
            "pros",
            "cons",
            "versus",
            "vs",
            "differ",
        ],
        trigger_types=[ProblemType.REASONING, ProblemType.BUSINESS, ProblemType.FINANCIAL],
        trigger_moves=[ReasoningMove.COMPARE],
        template="Option A pros/cons → Option B pros/cons → decisive factor → choose.",
        weight=1.1,
    ),
    ReasoningModule(
        id="INVERT",
        name="Inversion",
        trigger_keywords=[
            "impossible",
            "cannot",
            "never",
            "contradiction",
            "contrapositive",
            "prove by",
        ],
        trigger_types=[ProblemType.LOGIC, ProblemType.MATH],
        trigger_moves=[ReasoningMove.REFRAME, ReasoningMove.ELIMINATE],
        template="Assume negation → derive contradiction → original holds.",
        weight=1.2,
    ),
    ReasoningModule(
        id="CAUSAL",
        name="Causal Chain",
        trigger_keywords=[
            "because",
            "causes",
            "leads to",
            "results in",
            "effect",
            "consequence",
            "due to",
            "reason",
            "mechanism",
            "why does",
        ],
        trigger_types=[ProblemType.SCIENCE, ProblemType.REASONING, ProblemType.MEDICAL],
        trigger_moves=[ReasoningMove.INFER],
        template="Cause → mechanism → effect.",
        weight=1.2,
    ),
    ReasoningModule(
        id="ENUM",
        name="Enumerate Cases",
        trigger_keywords=[
            "all possible",
            "cases",
            "exhaustive",
            "try each",
            "enumerate",
            "for each",
            "systematic",
        ],
        trigger_types=[ProblemType.MATH, ProblemType.LOGIC],
        trigger_moves=[ReasoningMove.SPECIALIZE, ReasoningMove.ELIMINATE],
        template="List all cases → test each → collect valid ones.",
        weight=1.2,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HEURISTIC DETECTORS (v2 — refined)
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-compiled disambiguation patterns
_CODE_STRONG = re.compile(
    r"(def\s+\w+|class\s+\w+|import\s+\w+|for\s+\w+\s+in\s+|"
    r"while\s+\w+|print\s*\(|return\s+|if\s+__name__|"
    r"\w+\s*=\s*\[|\w+\.\w+\(|#include|void\s+\w+|"
    r"console\.\w+|function\s+\w+|const\s+\w+|let\s+\w+)",
    re.IGNORECASE,
)

_MATH_EXPRESSION = re.compile(r"\b\d+\s*[\+\-\*/]\s*\d+")
_MATH_WORDS = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|"
    r"sum|product|probability|area|volume|perimeter|factor|prime|"
    r"divisible|remainder|modulo|logarithm|exponent|polynomial|"
    r"inequality|fraction|decimal|percentage)\b",
    re.IGNORECASE,
)

_SCIENCE_WORDS = re.compile(
    r"\b(physics|chemistry|biology|experiment|hypothesis|theory|"
    r"molecule|atom|cell|force|energy|mass|velocity|acceleration|"
    r"evolution|quantum|relativity|thermodynamic|electromagnetic|"
    r"density|pressure|temperature|reaction|compound|element|"
    r"gravity|photosynthesis|mitosis|DNA|RNA|protein|enzyme|"
    r"ecosystem|climate|erosion|tectonic|magnetic|electric)\b",
    re.IGNORECASE,
)


def detect_problem_types(prompt: str) -> list[tuple[ProblemType, float]]:
    """Classify prompt into problem types with confidence scores (v2)."""
    prompt_lower = prompt.lower()
    scores: dict[ProblemType, float] = {pt: 0.0 for pt in ProblemType}

    # ── CODE detection — require strong signals ──
    code_strong_hits = len(_CODE_STRONG.findall(prompt))
    if code_strong_hits >= 1:
        scores[ProblemType.CODE] += 0.6 + min(code_strong_hits * 0.2, 0.4)
    lang_hits = len(
        re.findall(
            r"\b(python|java|javascript|c\+\+|rust|sql|typescript|ruby|go|swift)\b", prompt_lower
        )
    )
    if lang_hits:
        scores[ProblemType.CODE] += 0.5
    code_context = len(
        re.findall(
            r"\b(code|program|function|algorithm|debug|compile|syntax|output|snippet)\b",
            prompt_lower,
        )
    )
    if code_context and code_strong_hits:  # only boost if there's actual code
        scores[ProblemType.CODE] += min(code_context * 0.15, 0.4)

    # ── MATH detection ──
    math_expr_hits = len(_MATH_EXPRESSION.findall(prompt))
    if math_expr_hits:
        scores[ProblemType.MATH] += 0.4 + min(math_expr_hits * 0.15, 0.4)
    math_word_hits = len(_MATH_WORDS.findall(prompt))
    if math_word_hits:
        scores[ProblemType.MATH] += min(math_word_hits * 0.25, 0.8)
    # Numeric density signals math
    numbers = re.findall(r"\b\d+\.?\d*\b", prompt)
    if len(numbers) >= 3:
        scores[ProblemType.MATH] += 0.3
    if re.search(r"how (many|much|far|long|old|fast)", prompt_lower):
        scores[ProblemType.MATH] += 0.4
    if re.search(r"[=<>≤≥±∑∏∫√π]", prompt):
        scores[ProblemType.MATH] += 0.4

    # ── LOGIC detection ──
    logic_signals = [
        (r"\b(logic|logical|deduce|infer|conclusion|premise|valid|invalid)\b", 0.7),
        (r"\b(if.*then|therefore|must be)\b", 0.4),
        (r"\b(true|false|truth|lie|always|never)\b", 0.35),
        (r"\b(puzzle|riddle|brain teaser)\b", 0.7),
        (r"\b(each|different|neither|both|either)\b.*\b(have|has|is|are)\b", 0.4),
        (r"\bdoesn\'t have\b", 0.5),
    ]
    for pat, w in logic_signals:
        if re.search(pat, prompt_lower):
            scores[ProblemType.LOGIC] += w

    # ── SCIENCE detection — careful not to confuse with CODE ──
    sci_hits = len(_SCIENCE_WORDS.findall(prompt))
    if sci_hits:
        scores[ProblemType.SCIENCE] += min(sci_hits * 0.3, 0.9)
    if re.search(r"\bwhy does\b|\bhow does\b.*\b(work|happen|occur)\b", prompt_lower):
        scores[ProblemType.SCIENCE] += 0.3

    # ── MEDICAL ──
    med_signals = re.findall(
        r"\b(diagnosis|symptom|treatment|patient|disease|condition|clinical|"
        r"medication|dosage|prognosis|pathology)\b",
        prompt_lower,
    )
    scores[ProblemType.MEDICAL] += min(len(med_signals) * 0.3, 0.9)

    # ── LEGAL ──
    legal_signals = re.findall(
        r"\b(law|legal|court|statute|regulation|jurisdiction|liability|"
        r"contract|plaintiff|defendant|verdict|precedent)\b",
        prompt_lower,
    )
    scores[ProblemType.LEGAL] += min(len(legal_signals) * 0.3, 0.9)

    # ── FINANCIAL ──
    fin_signals = re.findall(
        r"\b(invest|stock|bond|portfolio|revenue|profit|loss|roi|"
        r"interest rate|compound|depreciation|valuation|market cap)\b",
        prompt_lower,
    )
    scores[ProblemType.FINANCIAL] += min(len(fin_signals) * 0.25, 0.8)
    if re.search(r"\$\d+", prompt):
        scores[ProblemType.FINANCIAL] += 0.2

    # ── BUSINESS ──
    biz_signals = re.findall(
        r"\b(market|customer|strategy|competitive|startup|"
        r"business model|revenue model|stakeholder|product launch)\b",
        prompt_lower,
    )
    scores[ProblemType.BUSINESS] += min(len(biz_signals) * 0.25, 0.8)

    # ── CREATIVE ──
    if re.search(
        r"\b(write|story|poem|creative|imagine|design|brainstorm|"
        r"compose|invent|innovate)\b",
        prompt_lower,
    ):
        scores[ProblemType.CREATIVE] += 0.6

    # ── FACTUAL ──
    if re.search(
        r"\b(what is|who is|when did|where is|define|explain|"
        r"describe|list the)\b",
        prompt_lower,
    ):
        scores[ProblemType.FACTUAL] += 0.4

    # ── MULTI_STEP ──
    step_markers = len(
        re.findall(
            r"\b(then|next|after that|first|second|finally|step|also|and then)\b", prompt_lower
        )
    )
    if step_markers >= 2:
        scores[ProblemType.MULTI_STEP] += 0.6
    if prompt_lower.count("?") >= 2:
        scores[ProblemType.MULTI_STEP] += 0.4
    # Word problems are inherently multi-step
    if scores[ProblemType.MATH] > 0.3 and len(prompt.split()) > 20:
        scores[ProblemType.MULTI_STEP] += 0.3

    # ── Disambiguation: Science "float" vs Code "float" ──
    if scores[ProblemType.CODE] > 0 and scores[ProblemType.SCIENCE] > 0:
        if code_strong_hits == 0:  # no actual code present
            scores[ProblemType.CODE] *= 0.3

    # ── Fallback ──
    if max(scores.values()) < 0.3:
        scores[ProblemType.REASONING] = 0.5

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(pt, round(sc, 2)) for pt, sc in ranked if sc > 0.2]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def detect_reasoning_moves(cot: str) -> list[tuple[ReasoningMove, float, str]]:
    """Extract reasoning moves from target CoT with evidence."""
    cot_lower = cot.lower()
    found: list[tuple[ReasoningMove, float, str]] = []

    move_signals: dict[ReasoningMove, list[tuple[str, float]]] = {
        ReasoningMove.DECOMPOSE: [
            (r"(break|split|decompose|separate|divide).*into", 0.9),
            (r"(first|step 1|part [a-d1-4]|sub-problem)", 0.5),
            (r"(let\'s consider|let\'s look at) (each|the)", 0.5),
            (r"(segment|phase|stage) \d", 0.6),
        ],
        ReasoningMove.CALCULATE: [
            (r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", 0.9),
            (r"\d+\s*[×÷]\s*\d+", 0.8),
            (r"(calculate|compute|multiply|divide|add|subtract)", 0.7),
            (r"=\s*\d+", 0.5),
        ],
        ReasoningMove.COMPARE: [
            (r"(compare|versus|vs\.?|greater|less than)", 0.8),
            (r"(on the other hand|alternatively|in contrast|however)", 0.6),
            (r"(better|worse) than", 0.6),
        ],
        ReasoningMove.INFER: [
            (r"(therefore|thus|hence|so\b|implies|means that|conclude)", 0.8),
            (r"(it follows|we can see|this tells us|this means)", 0.7),
            (r"(must be|has to be|can only be)", 0.7),
        ],
        ReasoningMove.VERIFY: [
            (r"(check|verify|confirm|double.check|validate|sanity)", 0.9),
            (r"(let\'s make sure|does this make sense|is this correct)", 0.7),
            (r"[✓✗]", 0.8),
        ],
        ReasoningMove.HYPOTHESIZE: [
            (r"(suppose|assume|if we|let\'s say|hypothesis|what if)", 0.8),
            (r"(consider the case|imagine|let me try)", 0.6),
        ],
        ReasoningMove.ELIMINATE: [
            (r"(eliminate|rule out|can\'t be|impossible|not possible)", 0.9),
            (r"(this doesn\'t work|this fails|rejected)", 0.7),
            (r"(not integer|not valid|doesn\'t satisfy)", 0.8),
        ],
        ReasoningMove.BACKTRACK: [
            (r"(wait|actually|no,\s|hmm|let me reconsider|mistake)", 0.9),
            (r"(going back|rethink|that\'s wrong|correction)", 0.8),
        ],
        ReasoningMove.SYNTHESIZE: [
            (r"(combining|putting.*together|overall|in summary|synthesis)", 0.8),
            (r"(final answer|the answer is|result is)", 0.7),
        ],
        ReasoningMove.ANALOGIZE: [
            (r"(similar to|analogy|just as|reminds me of)", 0.7),
            (r"(think of it as|imagine it like|this is like)", 0.8),
        ],
        ReasoningMove.REFRAME: [
            (r"(another way|different perspective|reframe|rephrase)", 0.8),
            (r"(look at it differently|equivalently|in other words)", 0.6),
        ],
        ReasoningMove.CONSTRAIN: [
            (r"(constraint|given that|must be|condition|subject to)", 0.8),
            (r"(only if|as long as|provided that|doesn\'t have)", 0.6),
        ],
        ReasoningMove.GENERALIZE: [
            (r"(in general|pattern|always|for any|universally)", 0.7),
            (r"(general rule|principle|formula|law)", 0.6),
        ],
        ReasoningMove.SPECIALIZE: [
            (r"(in this case|specifically|for example|instance)", 0.5),
            (r"(applying to|substituting|plugging in|if o=\d)", 0.7),
        ],
        ReasoningMove.TRACE: [
            (r"(trace|tracing|line by line|step through)", 0.9),
            (r"(x\s*=|y\s*=|variable|the list is now|state)", 0.6),
            (r"(creates a|makes.*point|reference.*same)", 0.5),
        ],
    }

    for move, patterns in move_signals.items():
        best_score = 0.0
        best_evidence = ""
        total_hits = 0
        for pattern, weight in patterns:
            matches = list(re.finditer(pattern, cot_lower))
            if matches:
                total_hits += len(matches)
                score = weight
                if score > best_score:
                    best_score = score
                    m = matches[0]
                    start = max(0, m.start() - 20)
                    end = min(len(cot), m.end() + 30)
                    best_evidence = cot[start:end].strip()
        # Boost for multiple occurrences
        if total_hits > 1:
            best_score = min(best_score + total_hits * 0.05, 1.0)
        if best_score > 0.3:
            found.append((move, round(best_score, 2), best_evidence))

    found.sort(key=lambda x: -x[1])
    return found


@dataclass
class CotStructure:
    """Extracted logical skeleton of a target CoT."""

    num_steps: int = 0
    has_backtrack: bool = False
    has_verification: bool = False
    has_enumeration: bool = False
    key_transitions: list[str] = field(default_factory=list)
    key_computations: list[str] = field(default_factory=list)
    pivotal_insight: str = ""
    reasoning_density: float = 0.0
    avg_step_length: float = 0.0


def extract_cot_structure(cot: str, answer: str) -> CotStructure:
    """Analyze target CoT to extract its logical skeleton."""
    struct = CotStructure()

    # Better step counting — look for explicit and implicit boundaries
    explicit_steps = re.findall(
        r"(step \d|first[\s,]|second[\s,]|third[\s,]|next[\s,]|then[\s,]|"
        r"finally[\s,]|now[\s,]|segment \d|if o=\d)",
        cot.lower(),
    )
    # Also count sentence-level reasoning units
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", cot) if s.strip()]
    struct.num_steps = max(len(explicit_steps), len(sentences) // 2, 1)

    struct.has_backtrack = bool(
        re.search(r"(wait|actually|no,|hmm|let me reconsider|mistake|oops)", cot.lower())
    )

    struct.has_verification = bool(
        re.search(r"(check|verify|confirm|double.check|let\'s make sure|✓|✗)", cot.lower())
    )

    struct.has_enumeration = bool(
        re.search(r"(if o=\d|case \d|try.*=\d|for each|all possible)", cot.lower())
    )

    # Key transitions
    transition_pattern = re.compile(
        r"\b(therefore|thus|hence|so|because|since|however|but|consequently|implies|means)\b",
        re.IGNORECASE,
    )
    for sent in sentences:
        if transition_pattern.search(sent) and len(sent) > 15:
            struct.key_transitions.append(sent[:120])

    # Key computations
    comps = re.findall(r"[\d]+\.?\d*\s*[×÷\+\-\*\/]\s*[\d]+\.?\d*\s*=\s*[\d]+\.?\d*", cot)
    struct.key_computations = comps[:8]

    # Pivotal insight
    answer_lower = str(answer).lower().strip()[:50]
    for i, sent in enumerate(sentences):
        if answer_lower[:20] in sent.lower() and i > 0:
            struct.pivotal_insight = sentences[i - 1][:150]
            break
    if not struct.pivotal_insight and len(sentences) >= 2:
        struct.pivotal_insight = sentences[-2][:150]

    # Reasoning density
    reasoning_tokens = len(
        re.findall(
            r"\b(because|therefore|thus|hence|so|since|implies|means|if|then|"
            r"given|assume|let|suppose|consider|note|must|can\'t)\b",
            cot.lower(),
        )
    )
    total_words = len(cot.split())
    struct.reasoning_density = round(reasoning_tokens / max(total_words, 1), 3)
    struct.avg_step_length = round(total_words / max(struct.num_steps, 1), 1)

    return struct


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SELF-DISCOVER MODULE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════


def select_modules(
    problem_types: list[tuple[ProblemType, float]],
    reasoning_moves: list[tuple[ReasoningMove, float, str]],
    prompt: str,
    max_modules: int = 5,
) -> list[tuple[ReasoningModule, float]]:
    """Self-Discover Phase 1: Select relevant reasoning modules."""
    prompt_lower = prompt.lower()
    type_scores = {pt: sc for pt, sc in problem_types}
    active_moves = {mv: sc for mv, sc, _ in reasoning_moves}

    scored: list[tuple[ReasoningModule, float]] = []

    for module in MODULES:
        score = 0.0

        # Type match — weighted by detection confidence
        for mt in module.trigger_types:
            if mt in type_scores:
                score += type_scores[mt] * 0.3

        # Move match — weighted by detection confidence
        for mm in module.trigger_moves:
            if mm in active_moves:
                score += active_moves[mm] * 0.35

        # Keyword match against prompt
        kw_hits = sum(1 for kw in module.trigger_keywords if kw in prompt_lower)
        score += min(kw_hits * 0.15, 0.6)

        # Apply module weight
        score *= module.weight
        score = round(score, 3)

        if score > 0.15:
            scored.append((module, score))

    scored.sort(key=lambda x: -x[1])

    # Deduplicate similar modules (keep highest scoring)
    selected = []
    selected_moves = set()
    for module, sc in scored:
        move_set = frozenset(module.trigger_moves)
        # Allow module if it brings at least one new move
        if not move_set.issubset(selected_moves) or len(selected) < 2:
            selected.append((module, sc))
            selected_moves.update(move_set)
        if len(selected) >= max_modules:
            break

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COT COMPRESSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


def compress_cot(cot: str, max_words: int = 200) -> str:
    """
    Extract the logical skeleton of a CoT, preserving reasoning flow.
    Uses a sentence-scoring approach with positional + content signals.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cot) if s.strip()]
    if not sentences:
        return cot

    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= max_words:
        return cot  # already short enough

    scored: list[tuple[str, float, int]] = []
    n = len(sentences)

    for i, sent in enumerate(sentences):
        score = 0.0
        sent_lower = sent.lower()

        # Reasoning connectors (high value)
        if re.search(r"\b(therefore|thus|hence|because|since|implies|means)\b", sent_lower):
            score += 3.0

        # Computations (high value for math)
        comp_hits = len(re.findall(r"\d+\s*[×÷\+\-\*\/=]\s*\d+", sent))
        score += comp_hits * 1.5

        # Key structural markers
        if re.search(r"\b(first|step|key|important|note|given|constraint)\b", sent_lower):
            score += 1.5

        # Self-correction (high value — shows reasoning depth)
        if re.search(r"\b(wait|actually|hmm|mistake|let me)\b", sent_lower):
            score += 2.5

        # Verification markers
        if re.search(r"[✓✗]|check|verify|confirm", sent_lower):
            score += 2.0

        # Conclusion signals
        if re.search(r"\b(so|therefore|answer|result|final|total)\b", sent_lower):
            score += 2.0

        # Positional bias — first and last are important
        if i == 0:
            score += 2.5
        if i >= n - 2:
            score += 3.0
        # Slight boost for early sentences (context)
        if i < 3:
            score += 0.5

        # Penalty for very short or filler sentences
        word_count = len(sent.split())
        if word_count < 5:
            score *= 0.5

        scored.append((sent, score, i))

    # Greedy selection by score, respecting word budget
    scored.sort(key=lambda x: -x[1])
    selected_indices = set()
    budget = 0

    for sent, sc, idx in scored:
        wc = len(sent.split())
        if budget + wc <= max_words:
            selected_indices.add(idx)
            budget += wc

    # Reconstruct in original order
    result = [
        s
        for i, (s, _, _) in enumerate(
            [(sent, sc, idx) for sent, sc, idx in scored if idx in selected_indices]
        )
    ]
    # Actually reconstruct properly by original index
    indexed = [(idx, sent) for sent, sc, idx in scored if idx in selected_indices]
    indexed.sort(key=lambda x: x[0])
    return " ".join(sent for _, sent in indexed)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANCHOR COMPOSER
# ═══════════════════════════════════════════════════════════════════════════════


def choose_style(struct: CotStructure, types: list[tuple[ProblemType, float]]) -> AnchorStyle:
    """Choose optimal anchor style based on problem + CoT characteristics."""
    primary_type = types[0][0] if types else ProblemType.REASONING
    primary_score = types[0][1] if types else 0.5

    # Short, direct problems → telegraphic
    if struct.num_steps <= 3 and not struct.has_backtrack and struct.avg_step_length < 15:
        return AnchorStyle.TELEGRAPHIC

    # Math/Code with clear steps → structured
    if primary_type in (ProblemType.MATH, ProblemType.CODE) and not struct.has_backtrack:
        return AnchorStyle.STRUCTURED

    # Complex reasoning with backtracking → scaffolded
    if struct.has_backtrack or struct.reasoning_density > 0.08 or struct.has_enumeration:
        return AnchorStyle.SCAFFOLDED

    # Everything else → narrative
    return AnchorStyle.NARRATIVE


def compose_anchor(
    prompt: str,
    cot: str,
    answer: str,
    modules: list[tuple[ReasoningModule, float]],
    structure: CotStructure,
    style: AnchorStyle,
    problem_types: list[tuple[ProblemType, float]],
    moves: list[tuple[ReasoningMove, float, str]],
    max_words: int = 512,
) -> str:
    """Compose the final anchor string."""
    parts: list[str] = []

    primary_type = problem_types[0][0].name.lower() if problem_types else "reasoning"
    top_module_names = [m.name for m, _ in modules[:3]]
    move_names = [mv.value for mv, _, _ in moves]
    answer_str = str(answer).strip()

    # ── Skeleton extraction ──
    # Reserve words for framing
    frame_budget = 15 if style == AnchorStyle.TELEGRAPHIC else 30
    skeleton = compress_cot(cot, max_words=max_words - frame_budget)

    # ── Style-specific composition ──
    if style == AnchorStyle.TELEGRAPHIC:
        parts.append(skeleton)
        parts.append(f"→ {answer_str}")

    elif style == AnchorStyle.STRUCTURED:
        parts.append(f"Approach: {' → '.join(top_module_names)}.")
        # Label sentences with detected moves
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", skeleton) if s.strip()]
        move_cycle = move_names if move_names else ["reason"]
        for i, sent in enumerate(sents):
            label = move_cycle[min(i, len(move_cycle) - 1)]
            parts.append(f"[{label}] {sent}")
        parts.append(f"[answer] {answer_str}")

    elif style == AnchorStyle.NARRATIVE:
        parts.append(f"Using {', '.join(top_module_names)}:")
        parts.append(skeleton)
        parts.append(f"The answer is {answer_str}.")

    elif style == AnchorStyle.SCAFFOLDED:
        parts.append(f"This {primary_type} problem needs: {', '.join(top_module_names)}.")
        if structure.has_enumeration:
            parts.append("[enumerate]")
        if structure.has_backtrack:
            parts.append("[explore→correct]")
        parts.append(skeleton)
        if structure.has_verification:
            parts.append("[verified]")
        parts.append(f"∴ {answer_str}")

    # ── Final trim ──
    anchor = " ".join(parts)
    words = anchor.split()
    if len(words) > max_words:
        words = words[: max_words - 3] + ["→", answer_str]
        anchor = " ".join(words)

    return anchor.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN API
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnchorResult:
    """Complete output from the anchor generator."""

    anchor: str
    style: str
    problem_types: list[str]
    selected_modules: list[str]
    reasoning_moves: list[str]
    cot_steps: int
    has_backtrack: bool
    has_verification: bool
    reasoning_density: float
    fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def generate_anchor(
    prompt: str,
    target_cot: str,
    answer: str,
    max_words: int = 512,
    max_modules: int = 5,
    style_override: str | None = None,
) -> AnchorResult:
    """
    Generate a reasoning anchor from (prompt, target_cot, answer).

    This is the single entry point. The anchor is a compact string that:
    1. Frames the problem type
    2. Names the reasoning strategy
    3. Preserves the essential logical skeleton of the CoT
    4. Terminates at the verifiable answer

    Designed as the target for RL fine-tuning — no loss, no turns,
    just a single string the model learns to follow.
    """
    # Phase 1: DETECT problem type
    problem_types = detect_problem_types(prompt)

    # Phase 2: EXTRACT reasoning moves from CoT
    reasoning_moves = detect_reasoning_moves(target_cot)

    # Phase 3: SELECT modules (Self-Discover Phase 1)
    modules = select_modules(problem_types, reasoning_moves, prompt, max_modules)

    # Phase 4: EXTRACT CoT structure
    structure = extract_cot_structure(target_cot, answer)

    # Phase 5: Choose composition style
    if style_override:
        style = AnchorStyle(style_override)
    else:
        style = choose_style(structure, problem_types)

    # Phase 6: COMPOSE anchor (Self-Discover Phase 2+3)
    anchor = compose_anchor(
        prompt=prompt,
        cot=target_cot,
        answer=answer,
        modules=modules,
        structure=structure,
        style=style,
        problem_types=problem_types,
        moves=reasoning_moves,
        max_words=max_words,
    )

    # Deterministic fingerprint
    fp_input = f"{prompt[:200]}|{answer}|{anchor[:100]}"
    fingerprint = hashlib.md5(fp_input.encode()).hexdigest()[:12]

    return AnchorResult(
        anchor=anchor,
        style=style.value,
        problem_types=[pt.name for pt, _ in problem_types[:3]],
        selected_modules=[m.id for m, _ in modules],
        reasoning_moves=[mv.value for mv, _, _ in reasoning_moves[:5]],
        cot_steps=structure.num_steps,
        has_backtrack=structure.has_backtrack,
        has_verification=structure.has_verification,
        reasoning_density=structure.reasoning_density,
        fingerprint=fingerprint,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def process_jsonl(
    input_path: str,
    output_path: str,
    prompt_key: str = "prompt",
    cot_key: str = "cot",
    answer_key: str = "answer",
    max_words: int = 512,
    anchor_only: bool = False,
):
    """
    Process a JSONL file of examples and write anchors.

    Input JSONL format: {"prompt": "...", "cot": "...", "answer": "..."}
    Output JSONL format: {"prompt": "...", "anchor": "...", "answer": "...", ...}
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    processed = 0
    errors = 0

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                result = generate_anchor(
                    prompt=ex[prompt_key],
                    target_cot=ex[cot_key],
                    answer=str(ex[answer_key]),
                    max_words=max_words,
                )

                if anchor_only:
                    out = {
                        "prompt": ex[prompt_key],
                        "output": result.anchor,
                    }
                else:
                    out = {
                        "prompt": ex[prompt_key],
                        "anchor": result.anchor,
                        "answer": str(ex[answer_key]),
                        "meta": {
                            "style": result.style,
                            "modules": result.selected_modules,
                            "moves": result.reasoning_moves,
                            "types": result.problem_types,
                            "steps": result.cot_steps,
                            "fp": result.fingerprint,
                        },
                    }

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1

            except Exception as e:
                errors += 1
                print(f"  [WARN] Line {line_num}: {e}", file=sys.stderr)

    print(f"Processed: {processed} | Errors: {errors} | Output: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DEMO & CLI
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_EXAMPLES = [
    {
        "name": "Math (word problem)",
        "prompt": "If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance?",
        "cot": "First, I need to calculate the distance for each segment. "
        "Distance = speed × time. "
        "Segment 1: 60 mph × 2.5 hours = 150 miles. "
        "Segment 2: 80 mph × 1.5 hours = 120 miles. "
        "Total distance = 150 + 120 = 270 miles.",
        "answer": "270 miles",
    },
    {
        "name": "Logic (constraint satisfaction)",
        "prompt": "Alice, Bob, and Carol each have a different pet: cat, dog, fish. "
        "Alice doesn't have the cat. Bob doesn't have the dog or the fish. "
        "Who has which pet?",
        "cot": "Let me work through the constraints. "
        "Bob doesn't have the dog or fish, so Bob must have the cat. "
        "Alice doesn't have the cat (and Bob has it), so Alice has either dog or fish. "
        "Since Bob has the cat, Carol has either dog or fish. "
        "Actually, we need more info. Wait, with Bob having the cat, "
        "and no further constraints distinguishing Alice and Carol... "
        "But typically the intended answer is Alice=dog, Carol=fish.",
        "answer": "Bob=cat, Alice=dog, Carol=fish",
    },
    {
        "name": "Code (reference semantics)",
        "prompt": "What does this Python code output?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))",
        "cot": "Let me trace through this code. "
        "x = [1, 2, 3] creates a list. "
        "y = x makes y point to the same list object (not a copy). "
        "y.append(4) adds 4 to the list, so the list is now [1, 2, 3, 4]. "
        "Since x and y reference the same list, x is also [1, 2, 3, 4]. "
        "len(x) = 4. So print outputs 4.",
        "answer": "4",
    },
    {
        "name": "Science (causal explanation)",
        "prompt": "Why does ice float on water?",
        "cot": "Ice floats because solid water is less dense than liquid water. "
        "This is unusual — most solids are denser than their liquid form. "
        "The reason is hydrogen bonding. In liquid water, hydrogen bonds constantly "
        "break and reform, allowing molecules to be relatively close. "
        "When water freezes, hydrogen bonds form a fixed crystalline lattice structure "
        "that spaces molecules further apart than in liquid water. "
        "This means ice has a lower density (~0.917 g/cm³) than liquid water (1.0 g/cm³). "
        "Since ice is less dense, it floats by Archimedes' principle.",
        "answer": "Ice is less dense than liquid water due to hydrogen bonding creating an open crystalline lattice structure",
    },
    {
        "name": "Math (enumeration + constraint)",
        "prompt": "A store sells apples for $2 each and oranges for $3 each. "
        "Maria buys some fruits and pays exactly $17. "
        "She buys more apples than oranges. How many of each?",
        "cot": "Let a = apples, o = oranges. "
        "2a + 3o = 17, a > o, a,o positive integers. "
        "If o=1: 2a = 14, a = 7. Check: 7 > 1 ✓ "
        "If o=2: 2a = 11, a = 5.5 ✗ "
        "If o=3: 2a = 8, a = 4. Check: 4 > 3 ✓ "
        "If o=4: 2a = 5, a = 2.5 ✗ "
        "If o=5: 2a = 2, a = 1. Check: 1 > 5 ✗ "
        "Valid: (7,1) and (4,3). Verify: 14+3=17 ✓, 8+9=17 ✓.",
        "answer": "7 apples & 1 orange, or 4 apples & 3 oranges",
    },
    {
        "name": "Factual (direct recall)",
        "prompt": "What is the capital of France?",
        "cot": "The capital of France is Paris.",
        "answer": "Paris",
    },
]


def demo():
    """Run demonstration with diverse examples."""
    print("=" * 80)
    print("  ReasonAnchor v2.0 — Demo")
    print("=" * 80)

    for ex in DEMO_EXAMPLES:
        result = generate_anchor(
            prompt=ex["prompt"],
            target_cot=ex["cot"],
            answer=ex["answer"],
        )

        print(f"\n{'━' * 80}")
        print(f"  📝 {ex['name']}")
        print(f"{'━' * 80}")
        print(f"  Q: {ex['prompt'][:75]}{'...' if len(ex['prompt']) > 75 else ''}")
        print(f"  A: {ex['answer']}")
        print("  ─────────────────────────────────────")
        print(f"  Style:    {result.style}")
        print(f"  Types:    {result.problem_types}")
        print(f"  Modules:  {result.selected_modules}")
        print(f"  Moves:    {result.reasoning_moves}")
        print(f"  Steps:    {result.cot_steps}  |  Density: {result.reasoning_density}")
        print(f"  Backtrack:{result.has_backtrack}  |  Verify: {result.has_verification}")
        print(f"  FP:       {result.fingerprint}")
        print("\n  🎯 ANCHOR:")
        # Word-wrap the anchor for display
        anchor_words = result.anchor.split()
        line = "  "
        for w in anchor_words:
            if len(line) + len(w) + 1 > 78:
                print(line)
                line = "  " + w
            else:
                line += " " + w if line.strip() else "  " + w
        if line.strip():
            print(line)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="ReasonAnchor v2.0 — Heuristic Reasoning Anchor Generator"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo examples")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file")
    parser.add_argument("--prompt-key", default="prompt", help="Key for prompt in input JSONL")
    parser.add_argument("--cot-key", default="cot", help="Key for CoT in input JSONL")
    parser.add_argument("--answer-key", default="answer", help="Key for answer in input JSONL")
    parser.add_argument("--max-words", type=int, default=512, help="Max words per anchor")
    parser.add_argument(
        "--anchor-only",
        action="store_true",
        help="Output only prompt+anchor (for direct RL training)",
    )
    parser.add_argument(
        "--single",
        nargs=3,
        metavar=("PROMPT", "COT", "ANSWER"),
        help="Generate single anchor from command line args",
    )

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.single:
        result = generate_anchor(
            args.single[0], args.single[1], args.single[2], max_words=args.max_words
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif args.input and args.output:
        process_jsonl(
            input_path=args.input,
            output_path=args.output,
            prompt_key=args.prompt_key,
            cot_key=args.cot_key,
            answer_key=args.answer_key,
            max_words=args.max_words,
            anchor_only=args.anchor_only,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

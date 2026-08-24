import os
import re
import sys

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_IADR1_RL = os.path.normpath(os.path.join(_CUR_DIR, "..", "..", "IAD-R1-main", "train", "stage_rl"))
if _IADR1_RL not in sys.path:
    sys.path.insert(0, _IADR1_RL)

from reward_process import location_reward, type_reward

# ── consistency (format) patterns — STRICT, ANSWER-LAST, no think/comparison ──
# Yes: exactly <type>...</type><location>...</location><answer>Yes</answer>
_PATTERN_YES = re.compile(
    r"\s*<type>.*?</type>\s*"
    r"<location>.*?</location>\s*"
    r"<answer>\s*yes\s*</answer>\s*",
    re.IGNORECASE | re.DOTALL,
)
# No: exactly <answer>No</answer>
_PATTERN_NO = re.compile(
    r"\s*<answer>\s*no\s*</answer>\s*",
    re.IGNORECASE | re.DOTALL,
)


def _content(c):
    """Extract the generated text from a TRL completion (conversational or str)."""
    if isinstance(c, list):
        if not c:
            return ""
        first = c[0]
        return first.get("content", "") if isinstance(first, dict) else str(first)
    if isinstance(c, dict):
        return c.get("content", "")
    return c if isinstance(c, str) else str(c)


def _gt_answer(sol):
    m = re.search(r"<answer>(.*?)</answer>", sol)
    return (m.group(1).strip() if m else sol.strip()).lower()


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _gating_enabled():
    """ANSWER_GATING env var (default on). Off -> ungated additive scoring,
    identical to reward_c_qwen.accuracy_reward_c_qwen."""
    v = os.environ.get("ANSWER_GATING", "1").strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ValueError(
        f"Invalid ANSWER_GATING={v!r}; expected one of {sorted(_TRUTHY | _FALSY)}."
    )


def consistency_reward_c_qwen_answer_last(completions, solution, **kwargs):
    """C-format consistency reward (STRICT fullmatch, ANSWER-LAST, no think/comparison)."""
    rewards = []
    for c, sol in zip(completions, solution):
        content = _content(c)
        gt = _gt_answer(sol)
        if gt == "yes":
            ok = _PATTERN_YES.fullmatch(content)
        elif gt == "no":
            ok = _PATTERN_NO.fullmatch(content)
        else:
            ok = None
        rewards.append(1.0 if ok else 0.0)
    return rewards


def accuracy_reward_c_qwen_answer_last(completions, solution, **kwargs):
    """C-format accuracy reward. Gating follows ANSWER_GATING (see _gating_enabled)."""
    gating = _gating_enabled()
    rewards = []
    for c, sol in zip(completions, solution):
        content = _content(c)
        reward = 0.0
        try:
            gt = _gt_answer(sol)

            if gt == "no":
                m = re.search(r"<answer>(.*?)</answer>", content)
                if m and m.group(1).strip().lower() == "no":
                    reward = 1.0

            elif gt == "yes":
                m = re.search(r"<answer>(.*?)</answer>", content)
                answer_ok = bool(m and m.group(1).strip().lower() == "yes")

                if not (gating and not answer_ok):
                    total_reward = 0.0
                    max_reward = 2.0

                    gpt_type = re.search(r"<type>(.*?)</type>", content)
                    gt_type = re.search(r"<type>(.*?)</type>", sol)
                    if gpt_type and gt_type:
                        pred_t = gpt_type.group(1).strip().lower()
                        gt_t = gt_type.group(1).strip().lower()
                        calc = type_reward.AnomalyRewardCalculator()
                        total_reward += calc.compute_reward(pred_t, gt_t)

                    gpt_loc = re.search(r"<location>(.*?)</location>", content)
                    gt_loc = re.search(r"<location>(.*?)</location>", sol)
                    if gpt_loc and gt_loc:
                        total_reward += location_reward.map_location_to_region(
                            gpt_loc.group(1).strip().lower(),
                            gt_loc.group(1).strip().lower(),
                        )

                    reward = total_reward / max_reward + (1.0 if answer_ok else 0.0)
        except Exception:
            pass
        rewards.append(reward)
    return rewards

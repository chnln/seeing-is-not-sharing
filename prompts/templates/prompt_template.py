"""
Prompt templates for the MapTask interpretation-matching evaluation.

This module mirrors the style of `scripts/prompt_templates.py` from:
https://github.com/jaaack-wang/lvlms-overhearing

We use `string.Template` placeholders (e.g., `${context}`) so that different
experimental conditions can be instantiated by swapping in pre-made fillings.
"""

from __future__ import annotations

from string import Template


INTERPRETATION_MATCHING_SYSTEM_PROMPT_TEMPLATE = Template(
    """
You are a dialogue analysis expert. You are overhearing two participants doing a MapTask-style route navigation task.

MapTask Background:
- Each participant has their own map and they cannot see each other's map.
- One participant (the Giver) describes a route using named landmarks on their map.
- The other participant (the Follower) tries to follow the instructions on their own map.
- The two maps may differ (some landmarks may be missing, duplicated, or placed differently), so the two participants can end up with different personal interpretations even if the dialogue sounds smooth.

Task:
You will be given a dialogue context in which ONE target reference expression is marked with <<...>>.
Decide whether the two participants interpret that reference expression as pointing to the SAME specific landmark (interpretations match).

Guidelines:
- Use only the provided dialogue context and (if available) the provided map image(s).
- A match can be supported by clear evidence of successful grounding (e.g., consistent descriptions, confirmations, coherent subsequent navigation).
- If the expression indicates quantificational asking, or unspecified/unresolved grounding, treat it as NOT a match.

Output:
Answer with exactly one word: Yes or No.
- Yes = interpretations match.
- No = interpretations do not match.
Do not output anything else.

Information you can access in this instance:
- Dialogue text: ${text_access}
- Map information: ${map_access}
""".strip()
)


INTERPRETATION_MATCHING_USER_PROMPT_TEMPLATE = Template(
    """
Below is the specific information for this judgement.

Target reference expression:
${target_ref}

Dialogue context (the target RE is wrapped in << >>):
${context}

Maps: if map images are provided, they appear below.
""".strip()
)


def get_interpretation_matching_prompt_templates() -> tuple[Template, Template]:
    """
    Returns (system_prompt_template, user_prompt_template).

    Expected variables:
    - ${text_access}: a short description of the available dialogue context window
    - ${map_access}: a short description of available map image(s)
    - ${target_ref}: the target reference expression (plain text)
    - ${context}: the dialogue context string (contains <<...>> around target RE)
    - ${maps}: map content block (can be empty; may include one or two images)
    """

    return INTERPRETATION_MATCHING_SYSTEM_PROMPT_TEMPLATE, INTERPRETATION_MATCHING_USER_PROMPT_TEMPLATE

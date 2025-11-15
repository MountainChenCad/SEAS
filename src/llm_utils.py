#!/usr/bin/env python3
"""
LLM utility functions for parsing and processing LLM outputs.
Extracted from main_experiment.py for reuse across scripts.
"""

import re
from typing import List, Optional


def parse_llm_output_for_label(
    llm_response_text: str,
    class_names: List[str],
    open_ended_match: bool = False
) -> Optional[str]:
    """
    Parse LLM response to extract predicted class label.

    Priority:
    1. Last occurrence (model's final answer)
    2. Structured format "Predicted Target Class: ..."
    3. Open-ended matching

    Args:
        llm_response_text: Raw LLM output text
        class_names: List of valid class names
        open_ended_match: If True, use more lenient matching

    Returns:
        Predicted class name if found, None otherwise
    """
    if not llm_response_text:
        return None

    # Strategy: Find the LAST occurrence of any class name (that's the model's final answer)
    # This handles cases like "<think>...GlobalHawk...</think>\nF35" where F35 is the answer
    processed_response = llm_response_text.lower()
    chars_to_remove = ['.', ',', ':', '"', '\'', '：', '`', '[', ']', '『', '』',
                       '【', '】', '(', ')', '（', '）', '*', '<', '>', '/', '\\']

    # Find all occurrences of class names
    class_occurrences = []  # (class_name, last_position)
    for cn in class_names:
        processed_cn = cn.lower()
        for char_to_remove in chars_to_remove:
            processed_cn = processed_cn.replace(char_to_remove, "")

        if not processed_cn:
            continue

        # Find last occurrence of this class name
        last_pos = processed_response.rfind(processed_cn)
        if last_pos != -1:
            class_occurrences.append((cn, last_pos))

    # Return the class name that appears LAST in the response
    if class_occurrences:
        # Sort by position, return the one appearing last
        class_occurrences.sort(key=lambda x: x[1], reverse=True)
        return class_occurrences[0][0]

    # Fallback: Original structured match
    match = re.search(r"Predicted Target Class:\s*([^\n`]+)`?", llm_response_text, re.IGNORECASE)
    if match:
        extracted_name = match.group(1).strip().rstrip('.')  # Remove trailing period
        for cn in class_names:
            if cn.lower() == extracted_name.lower():
                return cn

    # Final fallback: open-ended matching
    if open_ended_match:
        processed_response_early = llm_response_text.lower().strip()

        # Remove common lead-in phrases if open-ended
        lead_ins = [
            "the predicted target class is",
            "predicted target class:",
            "my prediction is",
            "the target is",
            "i believe the target is"
        ]
        for lead_in in lead_ins:
            if processed_response_early.startswith(lead_in):
                processed_response_early = processed_response_early[len(lead_in):].strip()

        for cn in class_names:
            # Exact match or near exact match
            if re.search(r'\b' + re.escape(cn.lower()) + r'\b', processed_response_early):
                return cn

    return None

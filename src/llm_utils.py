#!/usr/bin/env python3
"""
LLM utility functions for parsing and processing LLM outputs.
Extracted from main_experiment.py for reuse across scripts.
"""

import re
from typing import List, Optional


def parse_zeroshot_output(response: str, class_names: List[str]) -> Optional[str]:
    """
    Parse zero-shot format model output.

    Args:
        response: Model generated text
        class_names: Candidate class names ["F22", "F35", ...]

    Returns:
        Parsed class name, or None
    """
    if not response:
        return None

    # Strategy 1: Match "Predicted Target Class: XXX"
    pattern = r"Predicted\s*Target\s*Class:\s*([A-Za-z0-9\-]+)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Validate against class names
        for cn in class_names:
            if cn.lower() == extracted.lower():
                return cn
            if cn.replace("-", "").lower() == extracted.replace("-", "").lower():
                return cn

    # Strategy 2: Find the LAST occurrence of class names (for CoT format)
    # The model's final decision is typically the last class mentioned
    last_pos = -1
    last_class = None

    for cn in class_names:
        # Find all occurrences and track the last one
        pattern = re.compile(re.escape(cn), re.IGNORECASE)
        matches = list(pattern.finditer(response))
        if matches:
            pos = matches[-1].start()  # Last occurrence
            if pos > last_pos:
                last_pos = pos
                last_class = cn

    if last_class:
        return last_class

    # Strategy 3: Partial match (if no exact match found)
    response_upper = response.upper()
    for cn in class_names:
        if cn.upper() in response_upper:
            return cn

    return None


def parse_llm_output_for_label(
    llm_response_text: str,
    class_names: List[str],
    open_ended_match: bool = False,
    prefer_answer_tag: bool = False
) -> Optional[str]:
    """
    Parse LLM response to extract predicted class label.

    Priority:
    0. <answer> tag format (if prefer_answer_tag=True)
    1. Last occurrence (model's final answer)
    2. Structured format "Predicted Target Class: ..."
    3. Open-ended matching

    Args:
        llm_response_text: Raw LLM output text
        class_names: List of valid class names
        open_ended_match: If True, use more lenient matching
        prefer_answer_tag: If True, prioritize <answer> tag parsing (for SEAS models)

    Returns:
        Predicted class name if found, None otherwise
    """
    if not llm_response_text:
        return None

    # Strategy 0: Parse <answer> tag (SEAS model format)
    if prefer_answer_tag:
        answer_match = re.search(r"<answer>\s*([^<]+?)\s*</answer>", llm_response_text, re.IGNORECASE)
        if answer_match:
            extracted = answer_match.group(1).strip()
            # Check if extracted content contains a valid class name
            for cn in class_names:
                if cn.lower() in extracted.lower():
                    return cn

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

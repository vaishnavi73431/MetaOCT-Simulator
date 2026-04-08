from typing import Any

def grade_easy(state: Any, result: Any) -> float:
    try:
        if hasattr(result, 'reward'):
            return max(0.0, min(1.0, float(result.reward)))
        elif isinstance(result, dict) and 'reward' in result:
            return max(0.0, min(1.0, float(result['reward'])))
        return 0.0
    except Exception:
        return 0.0

def grade_medium(state: Any, result: Any) -> float:
    try:
        if hasattr(result, 'reward'):
            return max(0.0, min(1.0, float(result.reward)))
        elif isinstance(result, dict) and 'reward' in result:
            return max(0.0, min(1.0, float(result['reward'])))
        return 0.0
    except Exception:
        return 0.0

def grade_hard(state: Any, result: Any) -> float:
    try:
        if hasattr(result, 'reward'):
            return max(0.0, min(1.0, float(result.reward)))
        elif isinstance(result, dict) and 'reward' in result:
            return max(0.0, min(1.0, float(result['reward'])))
        return 0.0
    except Exception:
        return 0.0

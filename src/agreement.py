# Decision Diamond logic

from src.profiler import ProfilerOutput
from src.validator import ValidatorOutput

def check_agreement(prof_out: ProfilerOutput, val_out: ValidatorOutput) -> str:
    P_THRESH = 0.85
    V_THRESH = 0.85

    p_ok = prof_out.deviation_score >= P_THRESH
    v_ok = val_out.confidence >= V_THRESH and val_out.verdict == "Valid"

    if p_ok and v_ok:
        return "AGREE_VALID"
    elif not p_ok or not v_ok:
        return "CONFLICT_OR_INVALID"
    return "AMBIGUOUS"  # Escalate to Judge Agent
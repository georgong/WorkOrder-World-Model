"""
Schema definitions for the inference pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ScheduleRecord(BaseModel):
    """Single assignment record from a schedule upload."""
    assignment_id: Optional[str] = None
    task_id: Optional[str] = None
    engineer_id: Optional[str] = None
    district: Optional[str] = None
    department: Optional[str] = None
    start_time: Optional[str] = None
    duration: Optional[float] = None

    class Config:
        extra = "allow"


# Column name mappings: user-friendly → internal
COLUMN_ALIASES = {
    # assignment_id
    "assignment_id": "assignment_id",
    "ASSIGNMENT_ID": "assignment_id",
    "Z_ASSIGNMENTKEY_CHAR": "assignment_id",
    "W6KEY": "assignment_id",
    # task_id
    "task_id": "task_id",
    "TASK_ID": "task_id",
    "TASK": "task_id",
    # engineer_id
    "engineer_id": "engineer_id",
    "ENGINEER_ID": "engineer_id",
    "ASSIGNEDENGINEERS": "engineer_id",
    # district
    "district": "district",
    "DISTRICT": "district",
    # department
    "department": "department",
    "DEPARTMENT": "department",
    # start_time
    "start_time": "start_time",
    "STARTTIME": "start_time",
    "START_TIME": "start_time",
    # duration
    "duration": "duration",
    "DURATION": "duration",
}


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw CSV/JSON column names to canonical names."""
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        canonical = COLUMN_ALIASES.get(key, key)
        if canonical not in out:  # first match wins
            out[canonical] = value
    return out

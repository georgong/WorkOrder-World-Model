from pydantic import BaseModel
from typing import List

class FeatureSchema(BaseModel):
    category_feature: List[str]
    numeric_feature: List[str]
    key_cols: List[str]
    time_feature: List[str]

task_schema = FeatureSchema(
    category_feature=["REGION", "CITY"],
    numeric_feature=["DURATION", "TASKNUMBER"],
    key_cols=["W6KEY", "STATUS", "TASKTYPE", "DISTRICT", "DEPARTMENT"],
    time_feature=["EARLYSTART", "TIMECREATED", "DUEDATE", "LATESTART", "OPENDATE", "DISPLAYDATE", "SCHEDULEDSTART", "SCHEDULEDFINISH", "METRICDATE", "SCHEDULECOMPLETIONTIME"]
)

engineer_schema = FeatureSchema(
    category_feature=["CALENDAR", "CITY", "COMPANY", "CREW", "BUSINESSUNIT", "ACTIVE", "ENGINEERTYPE"],
    numeric_feature=["EFFICIENCY"],
    key_cols=["NAME", "DISTRICT", "DEPARTMENT"],
    time_feature=["TIMECREATED", "TIMEMODIFIED"]
)

assignment_schema = FeatureSchema(
    category_feature=["TIMECREATED", "TIMEMODIFIED", "NONAVAILABILITYTYPE", "Z_CONTRACTOR_ENGR_IND"],
    numeric_feature=[],
    key_cols=["TASK", "ASSIGNEDENGINEERS"],
    time_feature=["STARTTIME", "FINISHTIME", "ASSIGNMENTS_COMPLETIONTIME"]
)

district_schema = FeatureSchema(
    category_feature=["NAME", "CITY", "POSTCODE", "Z_DISTRICT_ABBR", "Z_DISTRICT_NAME", "Z_DISTRICT_CATEGORY"],
    numeric_feature=[],
    key_cols=["W6KEY"],
    time_feature=[]
)
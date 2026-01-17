from pydantic import BaseModel
from typing import List

class FeatureSchema(BaseModel):
    category_feature: List[str]
    numeric_feature: List[str]
    key_cols: List[str]
    time_feature: List[str]
    primary_key: str

# Baseline: TASK node/table

task_schema = FeatureSchema(
    category_feature=["REGION", "CITY", "DISTRICT", "DEPARTMENT"],
    numeric_feature=["TASKNUMBER"],
    key_cols=["W6KEY"],
    primary_key="W6KEY",
    time_feature=["DUEDATE", "OPENDATE","SCHEDULEDSTART", "SCHEDULEDFINISH"]
)
# Baseline: ENGINEER / CREW
engineer_schema = FeatureSchema(
    category_feature=["DEPARTMENT", "BUSINESSUNIT", "ACTIVE", "ENGINEERTYPE"],
    numeric_feature=["EFFICIENCY", "TRAVELSPEED"],
    key_cols=["W6KEY"], #or maybe use name, but name not that unique
    primary_key="W6KEY",
    time_feature=["TIMECREATED"]
)

# baseline: assignment 
assignment_schema = FeatureSchema(
    category_feature=["ISCREWASSIGNMENT", "NONAVAILABILITYTYPE", "Z_CONTRACTOR_ENGR_IND"],
    numeric_feature=[], #do not choose one, there is only a duration and duration have only 1 value
    key_cols=["TASK", "ASSIGNEDENGINEERS"], #Z_ASSIGNMENTKEY_CHAR：unique_ratio ≈ 0.995
    primary_key="W6KEY",
    time_feature=["STARTTIME", "TIMECREATED"]
)

#baseline: disctrict
district_schema = FeatureSchema(
    category_feature=[ "NAME", "CITY", "REGIONPARENT", "Z_DISTRICT_CATEGORY", "Z_DISTRICT_NAME", "USEDINMOBILE",],
    numeric_feature=[],
    key_cols=["W6KEY"],
    primary_key="W6KEY",
    time_feature=[]
)
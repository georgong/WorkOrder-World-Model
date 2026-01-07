import os
for i in os.walk("data/raw_data"):
    for root_dir, dirs, files in [i]:
        for file in files:
            if file.endswith(".csv"):
                print(os.path.join(root_dir, file))

from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.process.utils.filter_raw_data import drop_sparse_columns
from src.process.utils.convert_columns import convert_with_schema
from src.process.utils.convert_columns import remove_outliers_by_spec

paths = [
    "data/raw_data/W6TASKS-14.csv",
    "data/raw_data/W6TASK_STATUSES-0.csv",
    "data/raw_data/W6TASKS-3.csv",
    "data/raw_data/W6TASKS-2.csv",
    "data/raw_data/W6TASKS-15.csv",
    "data/raw_data/W6TASKS-17.csv",
    "data/raw_data/W6EQUIPMENT-1.csv",
    "data/raw_data/W6TASKS-0.csv",
    "data/raw_data/W6ENGINEERS-0.csv",
    "data/raw_data/W6TASKS-1.csv",
    "data/raw_data/W6TASK_TYPES-0.csv",
    "data/raw_data/W6EQUIPMENT-0.csv",
    "data/raw_data/W6TASKS-16.csv",
    "data/raw_data/W6TASKS-12.csv",
    "data/raw_data/W6TASKS-5.csv",
    "data/raw_data/W6TASKS-4.csv",
    "data/raw_data/W6TASKS-13.csv",
    "data/raw_data/W6TASKS-11.csv",
    "data/raw_data/W6TASKS-6.csv",
    "data/raw_data/W6TASKS-7.csv",
    "data/raw_data/W6TASKS-10.csv",
    "data/raw_data/W6ASSIGNMENTS-22.csv",
    "data/raw_data/W6ASSIGNMENTS-20.csv",
    "data/raw_data/W6ASSIGNMENTS-21.csv",
    "data/raw_data/W6ASSIGNMENTS-19.csv",
    "data/raw_data/W6ASSIGNMENTS-18.csv",
    "data/raw_data/W6ASSIGNMENTS-8.csv",
    "data/raw_data/W6ASSIGNMENTS-9.csv",
    "data/raw_data/W6ASSIGNMENTS-4.csv",
    "data/raw_data/W6ASSIGNMENTS-16.csv",
    "data/raw_data/W6ASSIGNMENTS-17.csv",
    "data/raw_data/W6ASSIGNMENTS-5.csv",
    "data/raw_data/W6ASSIGNMENTS-7.csv",
    "data/raw_data/W6ASSIGNMENTS-15.csv",
    "data/raw_data/W6REGIONS-0.csv",
    "data/raw_data/W6ASSIGNMENTS-14.csv",
    "data/raw_data/W6ASSIGNMENTS-6.csv",
    "data/raw_data/W6ASSIGNMENTS-10.csv",
    "data/raw_data/W6ASSIGNMENTS-2.csv",
    "data/raw_data/W6ASSIGNMENTS-3.csv",
    "data/raw_data/W6ASSIGNMENTS-11.csv",
    "data/raw_data/W6ASSIGNMENTS-13.csv",
    "data/raw_data/W6ASSIGNMENTS-1.csv",
    "data/raw_data/W6ASSIGNMENTS-0.csv",
    "data/raw_data/W6ASSIGNMENTS-12.csv",
    "data/raw_data/W6TASKS-21.csv",
    "data/raw_data/W6DEPARTMENT-0.csv",
    "data/raw_data/W6TASKS-20.csv",
    "data/raw_data/W6TASKS-9.csv",
    "data/raw_data/W6TASKS-8.csv",
    "data/raw_data/W6TASKS-18.csv",
    "data/raw_data/W6DISTRICTS-0.csv",
    "data/raw_data/W6TASKS-19.csv",
]

def group_and_concat(paths: List[str]) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, List[Path]] = {}

    for p in paths:
        p = Path(p)
        stem = p.stem 
        prefix = stem.split("-")[0] 
        groups.setdefault(prefix, []).append(p)

    out: Dict[str, pd.DataFrame] = {}
    for prefix, files in groups.items():
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            dfs.append(df)
        out[prefix] = pd.concat(dfs, ignore_index=True)

    return out

grouped_dfs = group_and_concat(paths)


task_df = drop_sparse_columns(grouped_dfs["W6TASKS"], min_non_na_ratio=0.1)

dtype_tasks = {
    "W6KEY": "Int64",
    "REVISION": "Int64",

    "CREATEDBY": "string",
    "TIMECREATED": "datetime64[ns]",
    "MODIFIEDBY": "string",
    "TIMEMODIFIED": "datetime64[ns]",

    "CALLID": "string",         
    "TASKNUMBER": "Int64",      

    #"EARLYSTART": "datetime64[ns]", #we shouldn't use for prediction
    "DUEDATE": "datetime64[ns]",
    #"LATESTART": "datetime64[ns]", #we shouldn't use for prediction
    "OPENDATE": "datetime64[ns]",

    # priority, status, region, district, postcode
    "PRIORITY": "Int64",
    "STATUS": "Int64",          
    "REGION": "Int64",
    "DISTRICT": "Int64",
    "POSTCODE": "string",        # postcode stringg

    "TASKTYPE": "Int64",        
    #"DURATION": "Int64",   #we shouldn't use for prediction 
    "NUMBEROFREQUIREDENGINEERS": "Int64",
    "CRITICAL": "Int64",        

    "CITY": "string",
    "TASKSTATUSCONTEXT": "Int64",    
    "ISCREWTASK": "Int64",          
    "COUNTRYID": "string",         
    "ISSCHEDULED": "Int64",
    "REQUIREDCREWSIZE": "Int64",
    "INJEOPARDY": "Int64",
    "PINNED": "Int64",
    "JEOPARDYSTATE": "Int64",

    "DISPLAYSTATUS": "Int64",


    "BUSINESSUNIT": "string",
    "COMPANY": "string",
    "DEPARTMENT": "string",

    "USEGEOCODE": "Int64",
    "SEMPRAPREREQUISITESMET": "Int64",
    "ISLOCKED": "Int64",
    "SEMPRAEMAILSENT": "Int64",


    "DISPLAYDATE": "datetime64[ns]",
    "SEMPRAWORKMGMTMODDATE": "datetime64[ns]",


    "SEMPRAINTERRUPTFLAG": "string",     #  string id
    "SEMPRAORDERDESCRIPTION": "string",
    "SEMPRALOCATIONMAP1": "string",
    "SEMPRALOCATIONMAP2": "string",


    "SEMPRAFACILITYID": "string",
    "SEMPRASPECIALEQUIPMENTFLAG": "Int64",
    "SEMPRASUSPENDFLAG": "Int64",
    "SEMPRAREFERFLAG": "Int64",
    "SEMPRACPFACILITYTYPE": "string",
    "SEMPRACOSTCENTER": "string",
    "SEMPRASCHEDULINGHORIZON": "Float64",  # horizon


    "OPTIMIZEONDATE": "datetime64[ns]",
    "INTSTATUSNAME": "string",           # COMPLETED / Map 


    "CMMATERIALSFLAG": "Int64",
    "CMPERMITSFLAG": "Int64",
    "CMATTACHMENTSFLAG": "Int64",
    "OCRFLAGGING2MAN": "Int64",
    "CMVOLUME": "Float64",
    "SEMPRACPREAD": "Float64",
    "ZZNOTUSEDCMODORINTENSITY": "Float64",
    "OCRGASCREW": "Int64",
    "OCRHAZMAT": "Int64",
    "CMCORROSION": "Int64",
    "OCRTREE": "Int64",
    "OCRUSAMARKOUT": "Int64",
    "CMREASONCODE": "Float64",          # Float64
    "CUSTOMERSAFFECTED": "Float64",
    "ZZNOTUSERCMSUPPORT": "Int64",
    "OCRENVIRONMENTAL": "Int64",
    "CMDELIVEREDFLAG": "Int64",
    "CMREADFLAG": "Int64",

    "SCHEDULEDSTART": "datetime64[ns]",
    "SCHEDULEDFINISH": "datetime64[ns]",
    #"ONSITETIMESTAMP": "datetime64[ns]", # we shouldn't use for prediction
    #"COMPLETIONTIMESTAMP": "datetime64[ns]", # we shouldn't use for prediction

    "FUNCTLOCREFNBR": "string",
    "OCRFLAGGING4MAN": "Int64",
    "OCRMACHINEDIGGER": "Int64",
    "CLICKPROJECTCODE": "string",

    "METRICDATE": "datetime64[ns]",
    "MAPORDER": "Int64",
    "DAYSFROMDUEDATE": "Float64",
    "DUEDATEBUFFER": "Float64",
    "DAYSESTODD": "Float64",

    "FL_FUNCTLOCDISP": "string",

    "SEMPRADISPATCHREADY": "Int64",
    "AMOPTOUT": "Int64",
    "UPLOADPENDINGFLAG": "Int64",

    "Z_TASKKEY_CHAR": "string",
    "Z_EARLYSTART_DATE": "datetime64[ns]",
    "Z_DUE_DATE": "datetime64[ns]",
    "Z_SCHEDULEDSTART_DATE": "datetime64[ns]",
    "Z_SCHEDULEDFINISH_DATE": "datetime64[ns]",
    "Z_TIMECREATED_DATE": "datetime64[ns]",
    "SEMPRAEMERGENCY": "Int64",        
}

task_df = convert_with_schema(
    task_df,
    dtype_tasks,
    year_min=2000,  
    year_max=2100,
    inplace=False,
    verbose=True,
)
task_df["SCHEDULECOMPLETIONTIME"] = task_df["SCHEDULEDFINISH"] - task_df["SCHEDULEDSTART"]
print(task_df.SCHEDULECOMPLETIONTIME.describe())
remove_outliers_by_spec(task_df,spec = 
    {
      "SCHEDULECOMPLETIONTIME": {"value_bounds": (0, None), "quantile_bounds": (None, None)},
    },combine="and",default_inclusive=True,default_keep_na=True,return_report=False)


import pandas as pd


def inspect_assignments_engineers(
    assignments_df: pd.DataFrame,
    engineers_df: pd.DataFrame,
    left_key: str = "ASSIGNEDENGINEERS",
    right_key: str = "NAME",
):
    """
    分析 assignments ↔ engineers 在 join key 上的关系。
    默认 ASSIGNEDENGINEERS ↔ NAME。
    """

    # 先 copy 一份，避免污染原 df
    left = assignments_df[[left_key]].copy()
    right = engineers_df[[right_key]].copy()

    # 统一成 string，strip 一下，空串 -> NA
    for df, col in [(left, left_key), (right, right_key)]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA,
                      "None": pd.NA, "NULL": pd.NA})
        )

    print("=== ASSIGNMENTS ↔ ENGINEERS 关系检查 ===")
    print(f"assignments 总行数: {len(left)}")
    print(f"engineers 总行数:   {len(right)}")

    # 唯一值数量
    n_left_keys = left[left_key].nunique(dropna=True)
    n_right_keys = right[right_key].nunique(dropna=True)
    print(f"\n左侧唯一 {left_key} 数: {n_left_keys}")
    print(f"右侧唯一 {right_key} 数: {n_right_keys}")

    # 左侧 key 出现次数（一个 engineer 有多少 assignment）
    vc_left = (
        left.dropna(subset=[left_key])
            [left_key]
            .value_counts()
    )

    if not vc_left.empty:
        max_assign_per_eng = int(vc_left.max())
        n_eng_multi_assign = int((vc_left > 1).sum())
        print(f"\n至少有 1 条 assignment 的工程师数: {len(vc_left)}")
        print(f"有多条 assignment 的工程师数: {n_eng_multi_assign}")
        print(f"单个工程师对应的 assignment 最大条数: {max_assign_per_eng}")
    else:
        print("\nASSIGNEDENGINEERS 全是空的，你这表是摆设吗。")

    # 右侧 key 统计（一个 NAME 是否重复多次）
    vc_right = (
        right.dropna(subset=[right_key])
             [right_key]
             .value_counts()
    )
    if not vc_right.empty:
        max_dup_name = int(vc_right.max())
        n_name_duplicated = int((vc_right > 1).sum())
        print(f"\nengineers.NAME 中出现重复的名字数: {n_name_duplicated}")
        print(f"同名最多重复次数: {max_dup_name}")
    else:
        print("\nengineers 里 NAME 基本是空的，这不科学。")

    # 检查引用完整性：有 assignment 找不到 engineer？有 engineer 从未被分配？
    left_keys = set(left[left_key].dropna().unique())
    right_keys = set(right[right_key].dropna().unique())

    only_in_assignments = left_keys - right_keys
    only_in_engineers = right_keys - left_keys

    print(f"\nASSIGNEDENGINEERS 中无对应 engineer 的 key 数: {len(only_in_assignments)}")
    print(f"从未被分配过 assignment 的 engineer 数: {len(only_in_engineers)}")

    # 简单判断关系类型（理论值：many-to-one, assignment→engineer）
    rel = []

    if (vc_left > 1).any():
        rel.append("many assignments → one engineer (多对一)")

    if (vc_right > 1).any():
        rel.append("one assignment key ↔ 多 engineer 记录（NAME 重复，多对多/脏数据嫌疑）")

    print("\n推测关系类型：")
    if not rel:
        print("  看起来更接近 1:1（但别太乐观，检查一下异常行数）")
    else:
        for r in rel:
            print("  -", r)

    # 给你看几个多 assignment 的工程师例子
    multi_assign = vc_left[vc_left > 1].head()
    if not multi_assign.empty:
        print("\n示例：有多条 assignment 的前几个工程师：")
        print(multi_assign)

def inspect_task_assignment_relation(task_df: pd.DataFrame,
                                     assignment_df: pd.DataFrame,
                                     task_key_col: str,
                                     assign_fk_col: str) -> dict:
    # 去掉 NaN，避免莫名其妙一堆 "nan 组"
    task_keys = task_df[task_key_col].dropna()
    assign_keys = assignment_df[assign_fk_col].dropna()

    # 每个 task 有多少条 assignment
    assign_counts = (
        assignment_df
        .dropna(subset=[assign_fk_col])
        .groupby(assign_fk_col)[assign_fk_col]
        .size()
        .rename("n_assignments")
    )

    max_assign_per_task = int(assign_counts.max()) if not assign_counts.empty else 0
    n_tasks_with_multi_assign = int((assign_counts > 1).sum())

    # 外键完整性：assignments.TASK 是否都能在 tasks.W6KEY 找到
    task_key_set    = set(task_keys.unique())
    assigns_task_set = set(assign_keys.unique())

    tasks_without_assign = task_key_set - assigns_task_set
    assigns_without_task = assigns_task_set - task_key_set

    summary = {
        "n_tasks": int(task_keys.nunique()),
        "n_assignments": int(assignment_df.shape[0]),
        "n_tasks_with_any_assign": int(assign_counts.shape[0]),
        "n_tasks_with_multi_assign": n_tasks_with_multi_assign,
        "max_assign_per_task": max_assign_per_task,
        "is_task_to_assign_one_to_one": (max_assign_per_task <= 1),
        "n_tasks_without_assignments": len(tasks_without_assign),
        "n_assign_task_values_without_task_row": len(assigns_without_task),
    }

    print("=== TASK ↔ ASSIGNMENT 关系检查 ===")
    print(f"tasks 唯一键数 ({task_key_col}): {summary['n_tasks']}")
    print(f"assignments 行数: {summary['n_assignments']}")
    print(f"有至少一条 assignment 的 task 数: {summary['n_tasks_with_any_assign']}")
    print(f"有多条 assignment 的 task 数: {summary['n_tasks_with_multi_assign']}")
    print(f"单个 task 对应的 assignment 最大条数: {summary['max_assign_per_task']}")
    print(f"是否 task→assignment 为 1:1: {summary['is_task_to_assign_one_to_one']}")
    print(f"没有任何 assignment 的 task 数: {summary['n_tasks_without_assignments']}")
    print(f"assignments.{assign_fk_col} 中无对应 task 的值个数: {summary['n_assign_task_values_without_task_row']}")

    if max_assign_per_task > 1:
        print("\n示例：有多条 assignment 的前几个 TASK：")
        print(assign_counts[assign_counts > 1].head(5))

    return summary

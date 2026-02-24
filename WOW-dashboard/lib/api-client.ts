/**
 * API client for the dashboard backend.
 */
import type { PredictResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/**
 * Upload all 7 required CSV datasets and run the full graph-based prediction pipeline.
 *
 * The backend expects these form-data field names:
 *   W6ASSIGNMENTS, W6DEPARTMENT, W6DISTRICTS, W6ENGINEERS,
 *   W6TASK_STATUSES, W6TASK_TYPES, W6TASKS
 */
export async function predictFromGraphFiles(
  files: Record<string, File>
): Promise<PredictResponse> {
  const formData = new FormData();

  // Map canonical filename → form field name expected by FastAPI
  const FILE_TO_FIELD: Record<string, string> = {
    "W6ASSIGNMENTS.csv": "W6ASSIGNMENTS",
    "W6DEPARTMENT.csv": "W6DEPARTMENT",
    "W6DISTRICTS.csv": "W6DISTRICTS",
    "W6ENGINEERS.csv": "W6ENGINEERS",
    "W6TASK_STATUSES.csv": "W6TASK_STATUSES",
    "W6TASK_TYPES.csv": "W6TASK_TYPES",
    "W6TASKS.csv": "W6TASKS",
  };

  for (const [canonicalName, fieldName] of Object.entries(FILE_TO_FIELD)) {
    const file = files[canonicalName];
    if (!file) {
      throw new Error(`Missing required file: ${canonicalName}`);
    }
    formData.append(fieldName, file, canonicalName);
  }

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Server error ${res.status}`);
  }

  return res.json();
}

/** Required file names the backend expects */
export const REQUIRED_FILES = [
  "W6ASSIGNMENTS.csv",
  "W6DEPARTMENT.csv",
  "W6DISTRICTS.csv",
  "W6ENGINEERS.csv",
  "W6TASK_STATUSES.csv",
  "W6TASK_TYPES.csv",
  "W6TASKS.csv",
] as const;

export type RequiredFileName = (typeof REQUIRED_FILES)[number];

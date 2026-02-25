import type { PredictResponse, GraphResponse} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

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

export async function fetchDemo(): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/demo`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Demo request failed (${res.status}): ${body}`);
  }
  return res.json();
}

export async function predictFromGraphFiles(
  files: Record<RequiredFileName, File>
): Promise<PredictResponse> {
  const form = new FormData();
  for (const [key, file] of Object.entries(files)) {
    // Strip ".csv" from the key to match the FastAPI parameter names
    const paramName = key.replace(".csv", "");
    form.append(paramName, file);
  }
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Predict request failed (${res.status}): ${body}`);
  }
  return res.json();
}

export async function fetchGraph(max_nodes: number = 300): Promise<GraphResponse> {
  const res = await fetch(`${API_BASE}/api/graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_nodes }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Graph request failed (${res.status}): ${body}`);
  }
  return res.json();
}

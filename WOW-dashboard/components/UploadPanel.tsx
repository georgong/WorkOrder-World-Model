"use client";

import { useCallback, useRef, useState } from "react";
import { predictFromGraphFiles, REQUIRED_FILES } from "@/lib/api";
import type { RequiredFileName } from "@/lib/api";
import type { PredictResponse, GraphResponse } from "@/lib/types";

interface Props {
  onResult: (data: PredictResponse, graph?: GraphResponse | null) => void;
  onDemo: () => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
  setError: (msg: string | null) => void;
}

/** Friendly display labels for each dataset */
const FILE_LABELS: Record<RequiredFileName, string> = {
  "W6ASSIGNMENTS.csv": "Assignments",
  "W6DEPARTMENT.csv": "Departments",
  "W6DISTRICTS.csv": "Districts",
  "W6ENGINEERS.csv": "Engineers",
  "W6TASK_STATUSES.csv": "Task Statuses",
  "W6TASK_TYPES.csv": "Task Types",
  "W6TASKS.csv": "Tasks",
};

export default function UploadPanel({
  onResult,
  onDemo,
  loading,
  setLoading,
  setError,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<Partial<Record<RequiredFileName, File>>>(
    {}
  );

  const uploadedCount = Object.keys(files).length;
  const allUploaded = uploadedCount === REQUIRED_FILES.length;

  /** Try to match a File to a required canonical name */
  const matchFile = (file: File): RequiredFileName | null => {
    const upper = file.name.toUpperCase();
    for (const req of REQUIRED_FILES) {
      if (upper === req.toUpperCase()) return req;
      const stem = req.replace(".csv", "").toUpperCase();
      if (upper.startsWith(stem)) return req;
    }
    return null;
  };

  const addFiles = useCallback(
    (incoming: FileList | File[]) => {
      setError(null);
      const next = { ...files };
      const unmatched: string[] = [];

      Array.from(incoming).forEach((file) => {
        if (!file.name.toLowerCase().endsWith(".csv")) {
          unmatched.push(`${file.name} (not a CSV)`);
          return;
        }
        const key = matchFile(file);
        if (key) {
          next[key] = file;
        } else {
          unmatched.push(file.name);
        }
      });

      setFiles(next);

      if (unmatched.length > 0) {
        setError(
          `Could not match: ${unmatched.join(", ")}. Expected filenames: ${REQUIRED_FILES.join(", ")}`
        );
      }
    },
    [files, setError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
      }
    },
    [addFiles]
  );

  const removeFile = (key: RequiredFileName) => {
    const next = { ...files };
    delete next[key];
    setFiles(next);
  };

  const handleSubmit = async () => {
    if (!allUploaded) return;
    setLoading(true);
    setError(null);

    try {
      const result = await predictFromGraphFiles(files as Record<RequiredFileName, File>);

      // Also request the serialized graph from the backend using the uploaded files
      const form = new FormData();
      for (const [key, file] of Object.entries(files as Record<RequiredFileName, File>)) {
        const paramName = key.replace(".csv", "");
        form.append(paramName, file);
      }
      // max_nodes is expected as a query param by the backend
      let graph: GraphResponse | null = null;
      try {
        const gr = await fetch(`/api/graph?max_nodes=300`, {
          method: "POST",
          body: form,
        });
        if (gr.ok) {
          graph = await gr.json();
        } else {
          const body = await gr.text();
          console.warn("Graph request failed:", gr.status, body);
        }
      } catch (err) {
        console.warn("Failed to fetch graph:", err);
      }

      onResult(result, graph);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFiles({});
    setError(null);
  };

  return (
    <div className="max-w-3xl mx-auto mt-8 mb-12 fade-in">
      {/* Header */}
      <div className="text-center mb-6">
        <h2 className="text-xl font-bold text-slate-800">
          WorkOrder Risk Analysis
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Upload all 7 required CSV files to build the graph and run risk
          analysis, or try a demo with synthetic data.
        </p>
      </div>

      {/* Demo CTA */}
      <div className="text-center mb-6">
        <button
          onClick={onDemo}
          disabled={loading}
          className={`
            inline-flex items-center gap-2 px-8 py-3 rounded-lg font-bold text-sm shadow-[0_4px_14px_0_rgba(1,21,139,0.39)] transition-all transform active:scale-95
            ${
              loading
                ? "bg-slate-200 text-slate-400 shadow-none cursor-not-allowed"
                : "bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600"
            }
          `}
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" />
              </svg>
              Loading demo data…
            </span>
          ) : (
            <>
              Try Demo Mode
            </>
          )}
        </button>
        <p className="text-xs text-slate-400 mt-2">
          Generates 200 synthetic work orders data
        </p>
      </div>

      {/* Divider */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex-1 h-px bg-slate-200" />
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
          or upload your data
        </span>
        <div className="flex-1 h-px bg-slate-200" />
      </div>

      {/* Drop zone */}
      <div
        className={`
          border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer group
          ${
            dragActive
              ? "border-brand-blue bg-brand-light"
              : "border-slate-200 bg-white hover:border-brand-blue hover:bg-slate-50"
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              addFiles(e.target.files);
            }
            e.target.value = "";
          }}
        />

        <div
          className={`w-12 h-12 mx-auto mb-3 rounded-full flex items-center justify-center transition-colors ${
            dragActive
              ? "bg-white text-brand-blue"
              : "bg-brand-light text-brand-blue group-hover:bg-brand-blue group-hover:text-white"
          }`}
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>

        <p className="text-sm font-semibold text-slate-700">
          Drag &amp; drop CSV files here, or click to browse
        </p>
        <p className="text-xs text-slate-400 mt-1">
          Select all 7 files at once, or add them one by one
        </p>
      </div>

      {/* File checklist */}
      <div className="mt-6 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-slate-600">Required Datasets</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${allUploaded ? "bg-green-100 text-green-700" : "bg-amber-100 text-amber-700"}`}>
              {uploadedCount} / {REQUIRED_FILES.length}
            </span>
          </div>
          {uploadedCount > 0 && (
            <button
              onClick={(e) => { e.stopPropagation(); handleReset(); }}
              className="text-[10px] font-semibold text-slate-400 hover:text-red-500 transition-colors"
            >
              Clear All
            </button>
          )}
        </div>

        <ul className="divide-y divide-slate-100">
          {REQUIRED_FILES.map((reqFile) => {
            const file = files[reqFile];
            const isUploaded = !!file;

            return (
              <li
                key={reqFile}
                className={`px-4 py-2.5 flex items-center justify-between transition-colors ${isUploaded ? "bg-green-50/40" : "bg-white"}`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${isUploaded ? "bg-green-500 text-white" : "bg-slate-200 text-slate-400"}`}>
                    {isUploaded ? (
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-700">{FILE_LABELS[reqFile]}</div>
                    <div className="text-[10px] text-slate-400 font-mono">{isUploaded ? file.name : reqFile}</div>
                  </div>
                </div>

                {isUploaded && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">{(file.size / 1024).toFixed(0)} KB</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeFile(reqFile); }}
                      className="text-slate-300 hover:text-red-500 transition-colors"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!allUploaded || loading}
        className={`
          mt-6 w-full py-3 rounded-lg font-bold text-sm shadow-md transition-all transform active:scale-95
          ${
            !allUploaded || loading
              ? "bg-slate-200 text-slate-400 shadow-none cursor-not-allowed"
              : "bg-brand-blue text-white hover:shadow-lg"
          }
        `}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
              <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" />
            </svg>
            Analyzing…
          </span>
        ) : (
          `Run Analysis (${uploadedCount}/${REQUIRED_FILES.length} files)`
        )}
      </button>
    </div>
  );
}

"use client";

import { useCallback, useRef, useState } from "react";
import { predictFromCSV, predictFromJSON } from "@/lib/api-client";
import type { PredictResponse } from "@/lib/types";

interface Props {
  onResult: (data: PredictResponse) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
  setError: (msg: string | null) => void;
}

export default function UploadPanel({
  onResult,
  loading,
  setLoading,
  setError,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [previewRows, setPreviewRows] = useState<Record<string, string>[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const parseCSV = (text: string): Record<string, string>[] => {
    const lines = text.trim().split("\n");
    if (lines.length < 2) return [];
    const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""));
    return lines.slice(1, 6).map((line) => {
      const values = line.split(",").map((v) => v.trim().replace(/"/g, ""));
      const row: Record<string, string> = {};
      headers.forEach((h, i) => {
        row[h] = values[i] || "";
      });
      return row;
    });
  };

  const handleFile = useCallback(
    (file: File) => {
      setError(null);
      if (!file.name.endsWith(".csv")) {
        setError("Please upload a .csv file");
        return;
      }
      setFileName(file.name);
      setSelectedFile(file);

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const rows = parseCSV(text);
        setPreviewRows(rows);
      };
      reader.readAsText(file);
    },
    [setError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const result = await predictFromCSV(selectedFile);
      onResult(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = async () => {
    setLoading(true);
    setError(null);

    // Generate sample records for demo
    const demoRecords = Array.from({ length: 50 }, (_, i) => ({
      assignment_id: `A${String(i + 1).padStart(4, "0")}`,
      task_id: `T${String(Math.floor(i / 2) + 1).padStart(4, "0")}`,
      engineer_id: `ENG${String((i % 10) + 1).padStart(3, "0")}`,
      district: `District ${(i % 5) + 1}`,
      department: `Dept ${(i % 3) + 1}`,
      start_time: new Date(
        2026,
        1,
        20 + Math.floor(i / 10),
        8 + (i % 8)
      ).toISOString(),
      duration: 1 + Math.random() * 10,
    }));

    try {
      const result = await predictFromJSON(demoRecords);
      onResult(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      {/* Upload area */}
      <div
        className={`
          border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer
          ${dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white hover:border-gray-400"}
        `}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />

        <div className="text-5xl mb-4">📄</div>
        <h2 className="text-xl font-semibold text-gray-700">
          Upload Schedule CSV
        </h2>
        <p className="text-gray-500 mt-2">
          Drag & drop a CSV file here, or click to browse
        </p>
        <p className="text-xs text-gray-400 mt-3">
          Expected columns: assignment_id, task_id, engineer_id, district,
          department, start_time, duration
        </p>
      </div>

      {/* File preview */}
      {fileName && (
        <div className="mt-6 bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              📎 {fileName}
            </span>
            <span className="text-xs text-gray-400">
              {previewRows.length > 0 ? `Showing first ${previewRows.length} rows` : ""}
            </span>
          </div>

          {previewRows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(previewRows[0]).map((col) => (
                      <th
                        key={col}
                        className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {previewRows.map((row, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="px-3 py-2 text-gray-600 whitespace-nowrap">
                          {String(val).substring(0, 30)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="mt-6 flex gap-4 justify-center">
        <button
          onClick={handleSubmit}
          disabled={!selectedFile || loading}
          className={`
            px-8 py-3 rounded-lg font-semibold text-white transition
            ${
              !selectedFile || loading
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 shadow-sm"
            }
          `}
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  className="opacity-25"
                />
                <path
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  className="opacity-75"
                />
              </svg>
              Processing…
            </span>
          ) : (
            "🚀 Analyze Schedule"
          )}
        </button>

        <button
          onClick={handleDemo}
          disabled={loading}
          className="px-8 py-3 rounded-lg font-semibold text-gray-700 bg-gray-100 hover:bg-gray-200 transition"
        >
          🧪 Run Demo
        </button>
      </div>
    </div>
  );
}

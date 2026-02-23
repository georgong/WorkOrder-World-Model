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
    <div className="max-w-2xl mx-auto mt-8 mb-12 fade-in">
      {/* Upload area */}
      <div
        className={`
          border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer group
          ${
            dragActive
              ? "border-brand-blue bg-brand-light"
              : "border-slate-200 bg-white hover:border-brand-blue hover:bg-slate-50"
          }
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

        <div className={`w-12 h-12 mx-auto mb-4 rounded-full flex items-center justify-center transition-colors ${dragActive ? 'bg-white text-brand-blue' : 'bg-brand-light text-brand-blue group-hover:bg-brand-blue group-hover:text-white'}`}>
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        
        <h2 className="text-lg font-bold text-slate-800">
          Upload Schedule CSV
        </h2>
        <p className="text-xs text-slate-500 mt-2 max-w-sm mx-auto">
          Drag & drop your schedule file here, or click to browse your computer
        </p>
        <div className="mt-6 flex gap-2 justify-center">
            {["assignment_id", "task_id", "engineer_id", "district"].map(col => (
               <span key={col} className="px-1.5 py-0.5 bg-slate-100 text-slate-500 text-[9px] uppercase font-bold rounded">
                 {col}
               </span>
            ))}
        </div>
      </div>

      {/* File preview */}
      {fileName && (
        <div className="mt-6 bg-white rounded-lg shadow-sm border border-slate-200 overflow-hidden">
          <div className="px-4 py-3 bg-slate-50 border-b border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-2">
               <div className="w-6 h-6 rounded bg-brand-light text-brand-blue flex items-center justify-center font-bold text-[10px]">CSV</div>
               <div>
                  <div className="text-xs font-bold text-slate-700">{fileName}</div>
                  <div className="text-[10px] text-slate-400">{previewRows.length} rows detected</div>
               </div>
            </div>
            
            <button 
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null); setFileName(null); }}
              className="text-slate-400 hover:text-red-500 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {previewRows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-[10px]">
                <thead className="bg-white border-b border-slate-100">
                  <tr>
                    {Object.keys(previewRows[0]).map((col) => (
                      <th
                        key={col}
                        className="px-4 py-2 text-left font-semibold text-slate-500 uppercase tracking-wider"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50 bg-slate-50/30">
                  {previewRows.map((row, i) => (
                    <tr key={i} className="hover:bg-indigo-50/30 transition-colors">
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="px-4 py-2 text-slate-600 whitespace-nowrap font-mono">
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
      <div className="mt-6 flex gap-3 justify-center">
        <button
          onClick={handleSubmit}
          disabled={!selectedFile || loading}
          className={`
            px-6 py-3 rounded-lg font-bold text-xs shadow-[0_4px_14px_0_rgba(1,21,139,0.39)] transition-all transform active:scale-95
            ${
              !selectedFile || loading
                ? "bg-slate-200 text-slate-400 shadow-none cursor-not-allowed"
                : "bg-brand-blue text-white hover:bg-brand-dark hover:-translate-y-0.5"
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
              Evaluating Risks...
            </span>
          ) : (
            "Analyze Schedule"
          )}
        </button>

        <button
          onClick={handleDemo}
          disabled={loading}
          className="px-6 py-3 rounded-lg font-bold text-xs text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:text-brand-blue hover:border-brand-blue/30 transition-all transform active:scale-95"
        >
          Try Demo Data
        </button>
      </div>
    </div>
  );
}

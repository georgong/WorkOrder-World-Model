"use client";

import { useMemo, useState } from "react";
import type { AssignmentPrediction } from "@/lib/types";

interface Props {
  predictions: AssignmentPrediction[];
}

type SortKey = "assignment_id" | "pred_completion_hours" | "risk_score";
type SortDir = "asc" | "desc";

function riskBadge(risk: number) {
  if (risk >= 0.7)
    return (
      <span className="inline-flex px-2 py-0.5 text-xs font-semibold rounded-full bg-red-100 text-red-700">
        High
      </span>
    );
  if (risk >= 0.4)
    return (
      <span className="inline-flex px-2 py-0.5 text-xs font-semibold rounded-full bg-amber-100 text-amber-700">
        Medium
      </span>
    );
  return (
    <span className="inline-flex px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-100 text-emerald-700">
      Low
    </span>
  );
}

export default function RiskTable({ predictions }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("risk_score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filter, setFilter] = useState<string>("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 20;

  const filtered = useMemo(() => {
    let data = [...predictions];
    if (filter) {
      const q = filter.toLowerCase();
      data = data.filter(
        (p) =>
          p.assignment_id.toLowerCase().includes(q) ||
          p.top_factors.some((f) => f.toLowerCase().includes(q))
      );
    }
    data.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      return sortDir === "asc"
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
    return data;
  }, [predictions, filter, sortKey, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sortIcon = (key: SortKey) => {
    if (sortKey !== key) return "↕️";
    return sortDir === "asc" ? "↑" : "↓";
  };

  const handleExport = () => {
    const headers = [
      "assignment_id",
      "pred_completion_hours",
      "risk_score",
      "risk_level",
      "top_factors",
    ];
    const rows = filtered.map((p) => [
      p.assignment_id,
      p.pred_completion_hours.toFixed(2),
      p.risk_score.toFixed(4),
      p.risk_score >= 0.7 ? "High" : p.risk_score >= 0.4 ? "Medium" : "Low",
      p.top_factors.join("; "),
    ]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "risk_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Toolbar */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search assignments..."
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setPage(0);
            }}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 w-64"
          />
          <span className="text-xs text-gray-500">
            {filtered.length} of {predictions.length} assignments
          </span>
        </div>
        <button
          onClick={handleExport}
          className="px-3 py-1.5 text-sm font-medium text-gray-600 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition"
        >
          Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th
                className="px-4 py-3 text-left font-medium text-gray-500 cursor-pointer hover:text-gray-700"
                onClick={() => toggleSort("assignment_id")}
              >
                Assignment {sortIcon("assignment_id")}
              </th>
              <th
                className="px-4 py-3 text-left font-medium text-gray-500 cursor-pointer hover:text-gray-700"
                onClick={() => toggleSort("pred_completion_hours")}
              >
                Predicted Hours {sortIcon("pred_completion_hours")}
              </th>
              <th
                className="px-4 py-3 text-left font-medium text-gray-500 cursor-pointer hover:text-gray-700"
                onClick={() => toggleSort("risk_score")}
              >
                Risk Score {sortIcon("risk_score")}
              </th>
              <th className="px-4 py-3 text-left font-medium text-gray-500">
                Risk Level
              </th>
              <th className="px-4 py-3 text-left font-medium text-gray-500">
                Top Factors
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {pageData.map((p) => (
              <tr key={p.assignment_id} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-mono text-xs">{p.assignment_id}</td>
                <td className="px-4 py-3">{p.pred_completion_hours.toFixed(2)}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-gray-200 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${
                          p.risk_score >= 0.7
                            ? "bg-red-500"
                            : p.risk_score >= 0.4
                            ? "bg-amber-500"
                            : "bg-emerald-500"
                        }`}
                        style={{ width: `${p.risk_score * 100}%` }}
                      />
                    </div>
                    <span className="text-xs">{(p.risk_score * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td className="px-4 py-3">{riskBadge(p.risk_score)}</td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {p.top_factors.map((f, i) => (
                      <span
                        key={i}
                        className="inline-flex px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
          <span className="text-xs text-gray-500">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
              className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
            >
              ← Prev
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(page + 1)}
              className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

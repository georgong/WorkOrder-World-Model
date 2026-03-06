"use client";

import type { ScheduleMetrics } from "@/lib/types";
import { useState } from "react";

interface Props {
  metrics: ScheduleMetrics;
}

function riskColor(value: number): string {
  if (value >= 0.7) return "text-red-600";
  if (value >= 0.4) return "text-orange-500";
  return "text-brand-green";
}

function riskBg(value: number): string {
  // Keeping white backgrounds for a clean look, using borders instead
  if (value >= 0.7) return "border-l-4 border-l-red-500";
  if (value >= 0.4) return "border-l-4 border-l-orange-400";
  return "border-l-4 border-l-brand-green";
}

function ProgressBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color =
    value / max >= 0.7
      ? "bg-red-500"
      : value / max >= 0.4
      ? "bg-orange-400"
      : "bg-brand-green";
  return (
    <div className="w-full bg-slate-100 rounded-full h-[6px] mt-2 overflow-hidden">
      <div
        className={`h-full rounded-full ${color} transition-all duration-1000 ease-out`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

const metricExplanations: Record<string, string> = {
  "Total Assignments":
    "The count of all work assignments included in the current schedule.",
  "Overall Risk Score":
    "Calculated as the proportion of assignments predicted to exceed 8 hours for completion. Higher means more assignments are at risk of being overdue.",
  "Workload Imbalance":
    "Measured using the Gini coefficient on the number of assignments per engineer. Higher values indicate a more uneven distribution.",
  "Avg Predicted Hours":
    "The mean predicted task completion time for all assignments, based on model inference.",
  "Median Predicted Hours":
    "The median predicted task completion time for all assignments, based on model inference.",
  "Most Overloaded Engineer":
    "The engineer assigned the most tasks in the current schedule.",
  "District with Highest Average Risk":
    "The district whose assignments have the highest average predicted risk score.",
  "Department with Highest Average Risk":
    "The department whose assignments have the highest average predicted risk score.",
};

function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative ml-1 select-none">
      <button
        type="button"
        aria-label="Info"
        className="align-top text-xs text-slate-500 hover:text-brand-blue focus:outline-none"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        tabIndex={0}
        style={{ lineHeight: 1 }}
      >
        ?
      </button>
      {open && (
        <span className="absolute z-10 left-1/2 -translate-x-1/2 top-6 min-w-[180px] bg-white border border-slate-200 shadow-lg rounded px-2 py-1 text-xs text-slate-700">
          {text}
        </span>
      )}
    </span>
  );
}

export default function MetricsCards({ metrics }: Props) {
  const cards = [
    {
      label: "Total Assignments",
      value: metrics.total_assignments,
      fmt: (v: number) => v.toLocaleString(),
      hasBar: false,
    },
    {
      label: "Overall Risk Score",
      value: metrics.overall_risk_score,
      fmt: (v: number) => (v * 100).toFixed(1) + "%",
      hasBar: true,
    },
    {
      label: "Workload Imbalance",
      value: metrics.workload_imbalance_score,
      fmt: (v: number) => (v * 100).toFixed(1) + "%",
      hasBar: true,
    },
    {
      label: "Avg Predicted Hours",
      value: metrics.avg_predicted_hours,
      fmt: (v: number) => v.toFixed(1) + "h",
      hasBar: false,
    },
    {
      label: "Median Predicted Hours",
      value: metrics.median_predicted_hours,
      fmt: (v: number) => v.toFixed(1) + "h",
      hasBar: false,
    },
    {
      label: "Most Overloaded Engineer",
      value: metrics.most_overloaded_engineer,
      fmt: (v: string) => v,
      hasBar: false,
    },
    {
      label: "District with Highest Average Risk",
      value: metrics.highest_risk_district,
      fmt: (v: string) => v,
      hasBar: false,
    },
    {
      label: "Department with Highest Average Risk",
      value: metrics.highest_risk_department,
      fmt: (v: string) => v,
      hasBar: false,
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
      {cards.map((c) => (
        <div
          key={c.label}
          className={`metric-card flex flex-col justify-between p-2 ${c.hasBar ? riskBg(Number(c.value)) : "border-l-4 border-l-brand-blue"}`}
        >
          <div className="flex items-start justify-between">
            <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block mb-0.5">
              {c.label}
            </span>
            <InfoTooltip text={metricExplanations[c.label]} />
          </div>
          <div
            className={`text-xl font-extrabold tracking-tight mt-0.5 ${
              c.hasBar ? riskColor(Number(c.value)) : "text-brand-dark"
            }`}
          >
            {c.fmt(c.value)}
          </div>
          {c.hasBar && <ProgressBar value={Number(c.value)} />}
        </div>
      ))}
    </div>
  );
}

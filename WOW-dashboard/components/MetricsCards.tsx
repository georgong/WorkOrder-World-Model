"use client";

import type { ScheduleMetrics } from "@/lib/types";

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
    <div className="w-full bg-slate-100 rounded-full h-1 mt-3 overflow-hidden">
      <div
        className={`h-full rounded-full ${color} transition-all duration-1000 ease-out`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function MetricsCards({ metrics }: Props) {
  const cards = [
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
      label: "Total Assignments",
      value: metrics.total_assignments,
      fmt: (v: number) => v.toLocaleString(),
      hasBar: false,
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
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className={`metric-card flex flex-col justify-between ${c.hasBar ? riskBg(c.value) : "border-l-4 border-l-brand-blue"}`}
        >
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-0.5">
              {c.label}
            </span>
            <div
              className={`text-2xl font-extrabold tracking-tight mt-0.5 ${
                c.hasBar ? riskColor(c.value) : "text-brand-dark"
              }`}
            >
              {c.fmt(c.value)}
            </div>
          </div>
          {c.hasBar && <ProgressBar value={c.value} />}
        </div>
      ))}
    </div>
  );
}

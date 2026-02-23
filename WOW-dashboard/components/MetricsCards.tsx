"use client";

import type { ScheduleMetrics } from "@/lib/types";

interface Props {
  metrics: ScheduleMetrics;
}

function riskColor(value: number): string {
  if (value >= 0.7) return "text-red-600";
  if (value >= 0.4) return "text-amber-600";
  return "text-emerald-600";
}

function riskBg(value: number): string {
  if (value >= 0.7) return "bg-red-50 border-red-200";
  if (value >= 0.4) return "bg-amber-50 border-amber-200";
  return "bg-emerald-50 border-emerald-200";
}

function ProgressBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color =
    value / max >= 0.7
      ? "bg-red-500"
      : value / max >= 0.4
      ? "bg-amber-500"
      : "bg-emerald-500";
  return (
    <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
      <div
        className={`h-2 rounded-full ${color} transition-all duration-500`}
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
      label: "Expected Overdue Rate",
      value: metrics.expected_overdue_rate,
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
      label: "Congestion Score",
      value: metrics.congestion_score,
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
          className={`metric-card ${c.hasBar ? riskBg(c.value) : ""}`}
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
              {c.label}
            </span>
          </div>
          <div
            className={`text-2xl font-bold ${
              c.hasBar ? riskColor(c.value) : "text-gray-900"
            }`}
          >
            {c.fmt(c.value)}
          </div>
          {c.hasBar && <ProgressBar value={c.value} />}
        </div>
      ))}
    </div>
  );
}

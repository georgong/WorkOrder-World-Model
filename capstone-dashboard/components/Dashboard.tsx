"use client";

import type { PredictResponse } from "@/lib/types";
import MetricsCards from "./MetricsCards";
import RiskTable from "./RiskTable";
import Charts from "./Charts";
import { useState } from "react";

interface Props {
  data: PredictResponse;
}

type Tab = "overview" | "table" | "charts";

export default function Dashboard({ data }: Props) {
  const [tab, setTab] = useState<Tab>("overview");

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: "overview", label: "Overview", icon: "📊" },
    { id: "table", label: "Assignments", icon: "📋" },
    { id: "charts", label: "Charts", icon: "📈" },
  ];

  return (
    <div>
      {/* Metadata bar */}
      <div className="mb-6 flex items-center gap-4 text-xs text-gray-500">
        <span>
          Mode: <strong>{String(data.metadata.mode || "demo")}</strong>
        </span>
        <span>•</span>
        <span>
          Runtime: <strong>{String(data.metadata.runtime_ms)}ms</strong>
        </span>
        <span>•</span>
        <span>
          Model: <strong>{String(data.metadata.model_version)}</strong>
        </span>
        {data.metadata.filename ? (
          <>
            <span>•</span>
            <span>
              File: <strong>{String(data.metadata.filename)}</strong>
            </span>
          </>
        ) : null}
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 mb-6 bg-gray-100 rounded-lg p-1 w-fit">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`
              px-4 py-2 rounded-md text-sm font-medium transition
              ${
                tab === t.id
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }
            `}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "overview" && (
        <div className="space-y-8">
          <MetricsCards metrics={data.schedule_metrics} />
          <Charts charts={data.charts} />
        </div>
      )}

      {tab === "table" && (
        <RiskTable predictions={data.assignment_predictions} />
      )}

      {tab === "charts" && <Charts charts={data.charts} fullWidth />}
    </div>
  );
}

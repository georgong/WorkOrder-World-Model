"use client";

import type { PredictResponse } from "@/lib/types";
import MetricsCards from "./MetricsCards";
import RiskTable from "./RiskTable";
import Charts from "./Charts";
import { useState } from "react";

interface Props {
  data: PredictResponse;
  onReset: () => void;
}

type Tab = "overview" | "table" | "charts";

export default function Dashboard({ data, onReset }: Props) {
  const [tab, setTab] = useState<Tab>("overview");

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "table", label: "Assignments" },
    { id: "charts", label: "Charts" },
  ];

  return (
    <div className="flex min-h-[calc(100vh-80px)] bg-slate-50 rounded-2xl overflow-hidden border border-slate-200 mt-4 shadow-sm">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
            Navigation
          </h2>
          <div className="space-y-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                  tab === t.id
                    ? "sidebar-link-active"
                    : "sidebar-link-inactive"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6 mt-auto bg-slate-50 border-t border-slate-200">
          <h3 className="text-xs font-semibold text-slate-500 mb-2">
            Schedule Info
          </h3>
          <div className="space-y-2 text-xs text-slate-500">
             <div className="flex justify-between">
              <span>Mode</span>
              <span className="font-mono text-slate-900">{String(data.metadata.mode || "demo")}</span>
            </div>
            <div className="flex justify-between">
              <span>Time</span>
              <span className="font-mono text-slate-900">{String(data.metadata.runtime_ms)}ms</span>
            </div>
            {data.metadata.filename && (
              <div className="pt-2 border-t border-slate-200 mt-2">
                 <span className="block truncate" title={String(data.metadata.filename)}>
                   {String(data.metadata.filename)}
                 </span>
              </div>
            )}
          </div>
          
          <button
            onClick={onReset}
            className="mt-6 w-full py-2 px-4 border border-slate-300 rounded-lg text-xs font-semibold text-slate-600 hover:bg-white hover:text-brand-blue transition-colors"
          >
           Upload New File
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto max-h-[calc(100vh-80px)]">
        <header className="mb-8">
          <h1 className="text-2xl font-bold text-slate-900">
            {tabs.find(t => t.id === tab)?.label}
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Analyze risk factors and schedule efficiency
          </p>
        </header>

        {tab === "overview" && (
          <div className="space-y-8 fade-in">
            <MetricsCards metrics={data.schedule_metrics} />
            <Charts charts={data.charts} />
          </div>
        )}

        {tab === "table" && (
          <div className="fade-in">
            <RiskTable predictions={data.assignment_predictions} />
          </div>
        )}

        {tab === "charts" && (
          <div className="fade-in">
            <Charts charts={data.charts} fullWidth />
          </div>
        )}
      </main>
    </div>
  );
}

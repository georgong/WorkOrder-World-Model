"use client";

import type { PredictResponse } from "@/lib/types";
import MetricsCards from "./MetricsCards";
import RiskTable from "./RiskTable";
import Charts from "./Charts";
import GraphVisualizer from "./GraphVisualizer";
import { useState } from "react";

interface Props {
  data: PredictResponse;
  onReset: () => void;
}

type Tab = "overview" | "table" | "charts" | "graph";

export default function Dashboard({ data, onReset }: Props) {
  const [tab, setTab] = useState<Tab>("overview");

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "table", label: "Assignments" },
    { id: "charts", label: "Charts" },
    { id: "graph", label: "Network Graph" },
  ];

  return (
    <div className="flex min-h-[calc(100vh-60px)] bg-slate-50 rounded-xl overflow-hidden border border-slate-200 mt-2 shadow-sm">
      {/* Sidebar */}
      <aside className="w-56 bg-white border-r border-slate-200 flex flex-col">
        <div className="p-4 border-b border-slate-100">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">
            Navigation
          </h2>
          <div className="space-y-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`w-full text-left px-3 py-2 rounded-md text-xs font-medium transition-all duration-200 ${
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

        <div className="p-4 mt-auto bg-slate-50 border-t border-slate-200">
          <h3 className="text-[10px] font-semibold text-slate-500 mb-2">
            Schedule Info
          </h3>
          <div className="space-y-1 text-[10px] text-slate-500">
             <div className="flex justify-between">
              <span>Mode</span>
              <span className="font-mono text-slate-900">{String(data.metadata.mode || "demo")}</span>
            </div>
            <div className="flex justify-between">
              <span>Time</span>
              <span className="font-mono text-slate-900">{String(data.metadata.runtime_ms)}ms</span>
            </div>
            {typeof data.metadata.filename === "string" && (
              <div className="pt-2 border-t border-slate-200 mt-2">
                 <span className="block truncate" title={data.metadata.filename}>
                   {data.metadata.filename}
                 </span>
              </div>
            )}
          </div>
          
          <button
            onClick={onReset}
            className="mt-4 w-full py-1.5 px-3 border border-slate-300 rounded-md text-[10px] font-semibold text-slate-600 hover:bg-white hover:text-brand-blue transition-colors"
          >
           Upload New File
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 overflow-y-auto max-h-[calc(100vh-60px)]">
        <header className="mb-6">
          <h1 className="text-xl font-bold text-slate-900">
            {tabs.find(t => t.id === tab)?.label}
          </h1>
          <p className="text-slate-500 text-xs mt-0.5">
            Analyze risk factors and schedule efficiency
          </p>
        </header>

        {tab === "overview" && (
          <div className="space-y-6 fade-in">
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

        {tab === "graph" && (
          <div className="fade-in">
            <div className="metric-card">
              <h3 className="text-xs font-bold text-slate-800 mb-4 uppercase tracking-wide">
                Assignment Risk Network
              </h3>
              <p className="text-[10px] text-slate-500 mb-4">
                Visualizing assignment nodes clustered by similarity. High risk nodes are red.
              </p>
              <GraphVisualizer predictions={data.assignment_predictions} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

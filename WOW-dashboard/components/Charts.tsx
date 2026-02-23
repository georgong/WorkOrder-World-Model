"use client";

import type { ChartData } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  charts: ChartData;
  fullWidth?: boolean;
}

function riskFill(value: number) {
  if (value >= 0.7) return "#ef4444";
  if (value >= 0.4) return "#fb923c"; // orange-400
  return "#58b83f"; // brand-green
}

export default function Charts({ charts, fullWidth }: Props) {
  const gridClass = fullWidth
    ? "grid grid-cols-1 gap-6"
    : "grid grid-cols-1 lg:grid-cols-2 gap-6";

  return (
    <div className={gridClass}>
      {/* Risk Histogram */}
      <div className="metric-card">
        <h3 className="text-xs font-bold text-slate-800 mb-4 uppercase tracking-wide">
          Risk Score Distribution
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={charts.risk_histogram}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="bin"
                tick={{ fontSize: 9, fill: '#64748b' }}
                angle={-30}
                textAnchor="end"
                height={40}
                axisLine={false} 
                tickLine={false}
              />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {charts.risk_histogram.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={riskFill((entry.binStart + entry.binEnd) / 2)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk by District */}
      <div className="metric-card">
        <h3 className="text-xs font-bold text-slate-800 mb-4 uppercase tracking-wide">
          Average Risk by District
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={charts.risk_by_district}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 9, fill: '#64748b' }}
                angle={-30}
                textAnchor="end"
                height={40}
                axisLine={false} 
                tickLine={false}
              />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} domain={[0, 1]} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(value: number) => [
                  `${(value * 100).toFixed(1)}%`,
                  "Avg Risk",
                ]}
              />
              <Bar dataKey="avg_risk" radius={[4, 4, 0, 0]}>
                {charts.risk_by_district.map((entry, i) => (
                  <Cell key={i} fill={riskFill(entry.avg_risk)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Workload by Engineer */}
      <div className="metric-card">
        <h3 className="text-xs font-bold text-slate-800 mb-4 uppercase tracking-wide">
          Workload by Engineer (Top 20)
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={charts.workload_by_engineer.slice(0, 20)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <YAxis
                dataKey="name"
                type="category"
                width={80}
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip cursor={{fill: '#f8fafc'}} />
              <Bar dataKey="assignments" fill="#01158b" radius={[0, 4, 4, 0]} barSize={12} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk by Department */}
      <div className="metric-card">
        <h3 className="text-xs font-bold text-slate-800 mb-4 uppercase tracking-wide">
          Average Risk by Department
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={charts.risk_by_department}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 9, fill: '#64748b' }}
                angle={-30}
                textAnchor="end"
                height={40}
                axisLine={false} 
                tickLine={false}
              />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} domain={[0, 1]} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(value: number) => [
                  `${(value * 100).toFixed(1)}%`,
                  "Avg Risk",
                ]}
              />
              <Bar dataKey="avg_risk" radius={[4, 4, 0, 0]}>
                {charts.risk_by_department.map((entry, i) => (
                  <Cell key={i} fill={riskFill(entry.avg_risk)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

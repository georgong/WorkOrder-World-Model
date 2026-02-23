"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import type { AssignmentPrediction } from "@/lib/types";

interface Props {
  predictions: AssignmentPrediction[];
}

interface Node extends d3.SimulationNodeDatum {
  id: string;
  group: string;
  risk: number;
}

export default function GraphVisualizer({ predictions }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Memoize nodes so simulation doesn't restart on every render unless data changes
  const nodes: Node[] = useMemo(() => {
    const map = new Map<string, Node>();
    predictions.forEach((p) => {
      if (!map.has(p.assignment_id)) {
        map.set(p.assignment_id, {
          id: p.assignment_id,
          group: "assignment",
          risk: p.risk_score,
        });
      }
    });
    return Array.from(map.values());
  }, [predictions]);

  useEffect(() => {
    if (!nodes.length || !svgRef.current || !wrapperRef.current) return;

    const width = wrapperRef.current.clientWidth || 800;
    const height = wrapperRef.current.clientHeight || 500;

    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height]);

    const g = svg.append("g");

    // Zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));

    svg.call(zoom);

    // Forces
    const simulation = d3
      .forceSimulation<Node>(nodes)
      .force("charge", d3.forceManyBody().strength(-20))
      .force("collide", d3.forceCollide().radius(8))
      .force("x", d3.forceX(width / 2).strength(0.08))
      .force("y", d3.forceY(height / 2).strength(0.08));

    // Render Nodes
    const circle = g
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 5)
      .attr("fill", (d) => {
        if (d.risk > 0.7) return "#ef4444";
        if (d.risk > 0.4) return "#fb923c";
        return "#58b83f";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
      });
    
    // Tooltip title
    circle.append("title").text((d) => `${d.id}\nRisk: ${(d.risk * 100).toFixed(0)}%`);

    simulation.on("tick", () => {
      circle.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes]);

  return (
    <div className="flex flex-col md:flex-row gap-4 h-[400px]">
      <div
        ref={wrapperRef}
        className="flex-1 bg-slate-50 border border-slate-100 rounded-lg overflow-hidden relative"
      >
        <svg ref={svgRef} className="w-full h-full block" />
        <div className="absolute top-2 left-2 bg-white/90 p-2 rounded border border-slate-100 shadow-sm">
           <div className="text-[10px] font-semibold text-slate-400 mb-1">Risk Levels</div>
           <div className="flex flex-col gap-1">
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500"></span><span className="text-[10px] text-slate-600">High (&gt;70%)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-orange-400"></span><span className="text-[10px] text-slate-600">Medium</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[#58b83f]"></span><span className="text-[10px] text-slate-600">Low</span></div>
           </div>
        </div>
      </div>

      {selectedNode && (
        <div className="w-full md:w-48 bg-white border border-slate-200 p-4 rounded-lg shadow-sm">
          <h3 className="text-xs font-bold text-slate-800 uppercase mb-2 border-b border-slate-100 pb-2">
            Details
          </h3>
          <div className="space-y-2 text-xs">
            <div>
              <span className="block text-slate-400 font-semibold">Assignment ID</span>
              <span className="font-mono text-slate-700">{selectedNode.id}</span>
            </div>
            <div>
              <span className="block text-slate-400 font-semibold">Risk Score</span>
              <span
                className={`font-bold ${
                  selectedNode.risk > 0.7
                    ? "text-red-500"
                    : selectedNode.risk > 0.4
                    ? "text-orange-400"
                    : "text-brand-green"
                }`}
              >
                {(selectedNode.risk * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

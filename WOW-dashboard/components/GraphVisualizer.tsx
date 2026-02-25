"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import type { AssignmentPrediction, GraphResponse } from "@/lib/types";

interface Props {
  predictions: AssignmentPrediction[];
  graph?: GraphResponse | null;
}

interface Node extends d3.SimulationNodeDatum {
  id: string;
  group: string; // node type
  risk?: number;
  label?: string;
}

interface Edge {
  source: string;
  target: string;
  type: string;
}

export default function GraphVisualizer({ predictions, graph }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [graphData, setGraphData] = useState<{
    nodes: any[];
    edges: any[];
  } | null>(null);

  // Accept graph prop from parent (set after Run Analysis); otherwise fall back to fetching on mount
  useEffect(() => {
    if (graph) {
      setGraphData({ nodes: graph.nodes || [], edges: graph.edges || [] });
      return;
    }
    // If no graph prop provided, do nothing here — component will use demo fallback
  }, [graph]);

  // Normalize and construct nodes/edges for D3. If graphData is null, fall back to demo extraction from predictions.
  const { nodes, edges } = useMemo(() => {
    console.log(graphData)
    if (graphData && graphData.nodes && graphData.edges) {
      // Create a quick lookup for prediction risks
      const riskMap = new Map<string, number>(predictions.map(p => [p.assignment_id, p.risk_score]));

      const normalizedNodes: Node[] = graphData.nodes.map((n) => {
        const id = n.id ?? n.node_id ?? n.name ?? String(n);
        const type = n.type ?? n.group ?? n.node_type ?? n.nodeType ?? "unknown";
        const label = n.label ?? n.name ?? id;
        const node: Node = { id, group: String(type), label };
        // If this looks like an assignment node, attach risk if available
        if (riskMap.has(id)) node.risk = riskMap.get(id);
        return node;
      });

      const normalizedEdges: Edge[] = graphData.edges.map((e) => {
        // e may have source/target as objects or ids; coerce to ids
        const s = e.source?.id ?? e.source?.node_id ?? e.source ?? e.src ?? e.u ?? null;
        const t = e.target?.id ?? e.target?.node_id ?? e.target ?? e.dst ?? e.v ?? null;
        return { source: String(s), target: String(t), type: e.type ?? e.rel ?? e.edge_type ?? "edge" } as Edge;
      }).filter(e => e.source && e.target);

      // Ensure all edges refer to nodes that are present in normalizedNodes
      const nodeIdSet = new Set(normalizedNodes.map((n) => n.id));
      const filteredEdges = normalizedEdges.filter((ee) => nodeIdSet.has(ee.source) && nodeIdSet.has(ee.target));

      return { nodes: normalizedNodes, edges: filteredEdges };
    }

    // Fallback demo graph (original behavior)
    const engineerSet = new Set<string>();
    const districtSet = new Set<string>();
    const departmentSet = new Set<string>();
    const assignmentNodes: Node[] = [];
    const engineerNodes: Node[] = [];
    const districtNodes: Node[] = [];
    const departmentNodes: Node[] = [];
    const edges: Edge[] = [];

    predictions.forEach((p) => {
      assignmentNodes.push({
        id: p.assignment_id,
        group: "assignment",
        risk: p.risk_score,
        label: p.assignment_id,
      });
      // Demo: try to extract engineer, district, department from top_factors
      let engineer = "";
      let district = "";
      let department = "";
      p.top_factors.forEach(f => {
        if (f.startsWith("ENG-")) engineer = f;
        else if (["North","South","East","West","Central","Downtown","Suburb-A","Suburb-B"].includes(f)) district = f;
        else if (["Electrical","Plumbing","HVAC","Structural","General Maintenance","Landscaping"].includes(f)) department = f;
      });
      if (engineer) engineerSet.add(engineer);
      if (district) districtSet.add(district);
      if (department) departmentSet.add(department);
      if (engineer) edges.push({ source: p.assignment_id, target: engineer, type: "assigned_to" });
      if (district) edges.push({ source: p.assignment_id, target: district, type: "in_district" });
      if (department) edges.push({ source: p.assignment_id, target: department, type: "in_department" });
    });
    engineerSet.forEach(e => engineerNodes.push({ id: e, group: "engineer", label: e }));
    districtSet.forEach(d => districtNodes.push({ id: d, group: "district", label: d }));
    departmentSet.forEach(dep => departmentNodes.push({ id: dep, group: "department", label: dep }));
    const nodes = [...assignmentNodes, ...engineerNodes, ...districtNodes, ...departmentNodes];
    return { nodes, edges };
  }, [graphData, predictions]);

  // Compute unique node types and assign colors (stable across renders)
  const uniqueTypes = useMemo(() => Array.from(new Set(nodes.map((n) => n.group))), [nodes]);
  const nodeColors: Record<string, string> = useMemo(() => {
    const palette = (d3.schemeTableau10 as string[]) || [
      "#2563eb",
      "#fb923c",
      "#38bdf8",
      "#a21caf",
      "#84cc16",
      "#f97316",
    ];
    const colorScale = d3.scaleOrdinal<string, string>(palette).domain(uniqueTypes as any);
    const mapping: Record<string, string> = {};
    uniqueTypes.forEach((t) => {
      mapping[t] = colorScale(t);
    });
    return mapping;
  }, [uniqueTypes]);

  useEffect(() => {
    if (!nodes.length || !svgRef.current || !wrapperRef.current) return;

    const width = wrapperRef.current.clientWidth || 800;
    const height = wrapperRef.current.clientHeight || 500;

    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height]);
    const g = svg.append("g");

    // Zoom
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // nodeColors and uniqueTypes are computed outside so legend and rendering share them

    // Edge rendering
    const edgeLines = g
      .append("g")
      .selectAll("line")
      .data(edges)
      .join("line")
      .attr("stroke", "#cbd5e1")
      .attr("stroke-width", 1.2);

    // Node rendering
    const circle = g
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => d.group === "assignment" ? 6 : 8)
      .attr("fill", (d) => nodeColors[d.group] || "#64748b")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
      });

    // Tooltip
    circle.append("title").text((d) => `${d.label || d.id}${d.risk !== undefined ? `\nRisk: ${(d.risk * 100).toFixed(0)}%` : ''}`);

    // Node labels (optional, for non-assignment nodes)
    const labelText = g
      .append("g")
      .selectAll("text")
      .data(nodes.filter(n => n.group !== "assignment"))
      .join("text")
      .attr("font-size", 10)
      .attr("fill", "#334155")
      .attr("text-anchor", "middle")
      .text((d) => d.label || d.id);

    // D3 force simulation
    const simulation = d3
      .forceSimulation<Node>(nodes)
      .force("charge", d3.forceManyBody().strength(-30))
      .force("collide", d3.forceCollide().radius(10))
      .force("link", d3.forceLink(edges).id((d: any) => d.id).distance(60).strength(0.7))
      .force("x", d3.forceX(width / 2).strength(0.08))
      .force("y", d3.forceY(height / 2).strength(0.08));

    simulation.on("tick", () => {
      edgeLines
        .attr("x1", (d) => (nodes.find(n => n.id === d.source)?.x) || 0)
        .attr("y1", (d) => (nodes.find(n => n.id === d.source)?.y) || 0)
        .attr("x2", (d) => (nodes.find(n => n.id === d.target)?.x) || 0)
        .attr("y2", (d) => (nodes.find(n => n.id === d.target)?.y) || 0);
      circle
        .attr("cx", (d) => d.x!)
        .attr("cy", (d) => d.y!);
      labelText
        .attr("x", (d) => d.x!)
        .attr("y", (d) => d.y! - 12);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges]);

  return (
    <div className="flex flex-col md:flex-row gap-4 h-[400px]">
      <div
        ref={wrapperRef}
        className="flex-1 bg-slate-50 border border-slate-100 rounded-lg overflow-hidden relative"
      >
        <svg ref={svgRef} className="w-full h-full block" />
          <div className="absolute top-2 left-2 bg-white/90 p-2 rounded border border-slate-100 shadow-sm">
            <div className="text-[10px] font-semibold text-slate-400 mb-1">Node Types</div>
            <div className="flex flex-col gap-1">
             {uniqueTypes.map((t) => (
               <div key={t} className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full" style={{ background: nodeColors[t] }} />
                <span className="text-[10px] text-slate-600">{t}</span>
               </div>
             ))}
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
              <span className="block text-slate-400 font-semibold">Node ID</span>
              <span className="font-mono text-slate-700">{selectedNode.id}</span>
            </div>
            <div>
              <span className="block text-slate-400 font-semibold">Type</span>
              <span className="font-mono text-slate-700">{selectedNode.group}</span>
            </div>
            {selectedNode.risk !== undefined && (
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
            )}
          </div>
        </div>
      )}
    </div>
  );
}

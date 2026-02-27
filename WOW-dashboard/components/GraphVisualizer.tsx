"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import type { AssignmentPrediction, GraphResponse } from "@/lib/types";

interface Props {
  predictions: AssignmentPrediction[];
  graph?: GraphResponse | null;
  // Optional Tailwind height class applied to the outer container (e.g. "h-[400px]" or "h-[calc(100vh-200px)]")
  heightClass?: string;
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

export default function GraphVisualizer({ predictions, graph, heightClass = "h-[400px]" }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  // Use a ref to hold the currently-selected node to avoid triggering
  // the main simulation effect (which would reset the view). Use a
  // small state `selectedInfo` for the details panel only.
  const selectedNodeRef = useRef<Node | null>(null);
  const [selectedInfo, setSelectedInfo] = useState<{ id: string; group: string; risk?: number } | null>(
    null
  );
  const nodesByIdRef = useRef<Map<string, Node> | null>(null);
  const gRef = useRef<any>(null);
  // Fullscreen state for the graph container
  const [isFullscreen, setIsFullscreen] = useState(false);
  // Request fullscreen for the wrapper element
  const enterFullscreen = async () => {
    const el = wrapperRef.current;
    if (!el) return;
    try {
      await (el as any).requestFullscreen();
    } catch (e) {
      // ignore failures
    }
  };

  // Exit fullscreen if active
  const exitFullscreen = async () => {
    if (document.fullscreenElement) {
      try {
        await (document as any).exitFullscreen();
      } catch (e) {
        // ignore
      }
    }
  };

  // Normalize and construct nodes/edges for D3.
  const { nodes, edges } = useMemo(() => {
    if (graph && graph.nodes && graph.edges && graph.nodes.length > 0) {
      const riskMap = new Map<string, number>(predictions.map((p) => [p.assignment_id, p.risk_score]));

      // Canonical mapping from backend plural types to singular group names used in demo
      const TYPE_MAP: Record<string, string> = {
        assignments: "assignment",
        engineers: "engineer",
        districts: "district",
        departments: "department",
      };

      // Only include these canonical groups in the visualization
      const ALLOWED = new Set(["assignment", "engineer", "district", "department", "task", "task_type", "task_statuses"]);
      console.log ("Graph nodes:", graph.nodes, "edges:", graph.edges);
      // Map backend nodes to canonical visualization nodes and filter by allowed types
      const normalizedNodes: Node[] = graph.nodes
        .map((n) => {
          const id = String(n.id);
          const rawType = String(n.type ?? n.node_type ?? n.group ?? "unknown");
          const type = TYPE_MAP[rawType] ?? rawType.replace(/s$/i, "");
          const label = String(n.label ?? n.name ?? id);
          // Preserve layout positions from backend if present (helps spread)
          const node: Node = { id, group: type, label, x: n.x ?? undefined, y: n.y ?? undefined } as Node;
          if (riskMap.has(label)) node.risk = riskMap.get(label);
          else if (riskMap.has(id)) node.risk = riskMap.get(id);
          return node;
        })
        .filter((n) => ALLOWED.has(n.group));

      // Build a set of node ids for fast filtering
      const nodeIdSet = new Set(normalizedNodes.map((n) => n.id));

      // Normalize edges and filter out any that reference missing nodes
      const normalizedEdges: Edge[] = (graph.edges || [])
        .map((e) => ({
          source: String(e.source),
          target: String(e.target),
          type: String(e.type ?? e.rel ?? e.edge_type ?? "edge"),
        }))
        .filter((ee) => nodeIdSet.has(ee.source) && nodeIdSet.has(ee.target));

      // Remove nodes that have no edges (keep only nodes that appear as source/target)
      const edgeNodeSet = new Set<string>(normalizedEdges.flatMap((e) => [e.source, e.target]));
      const filteredNodes = normalizedNodes.filter((n) => edgeNodeSet.has(n.id));

      return { nodes: filteredNodes, edges: normalizedEdges };
    }

    // Demo fallback (unchanged)
    const engineerSet = new Set<string>();
    const districtSet = new Set<string>();
    const departmentSet = new Set<string>();
    const assignmentNodes: Node[] = [];
    const engineerNodes: Node[] = [];
    const districtNodes: Node[] = [];
    const departmentNodes: Node[] = [];
    const edges: Edge[] = [];

    predictions.forEach((p) => {
      assignmentNodes.push({ id: p.assignment_id, group: "assignment", risk: p.risk_score, label: p.assignment_id });
      let engineer = "";
      let district = "";
      let department = "";
      p.top_factors.forEach((f) => {
        if (f.startsWith("ENG-")) engineer = f;
        else if (["North", "South", "East", "West", "Central", "Downtown", "Suburb-A", "Suburb-B"].includes(f)) district = f;
        else if (["Electrical", "Plumbing", "HVAC", "Structural", "General Maintenance", "Landscaping"].includes(f)) department = f;
      });
      if (engineer) engineerSet.add(engineer);
      if (district) districtSet.add(district);
      if (department) departmentSet.add(department);
      if (engineer) edges.push({ source: p.assignment_id, target: engineer, type: "assigned_to" });
      if (district) edges.push({ source: p.assignment_id, target: district, type: "in_district" });
      if (department) edges.push({ source: p.assignment_id, target: department, type: "in_department" });
    });
    engineerSet.forEach((e) => engineerNodes.push({ id: e, group: "engineer", label: e }));
    districtSet.forEach((d) => districtNodes.push({ id: d, group: "district", label: d }));
    departmentSet.forEach((dep) => departmentNodes.push({ id: dep, group: "department", label: dep }));
    const allNodes = [...assignmentNodes, ...engineerNodes, ...districtNodes, ...departmentNodes];
    // Filter out nodes that do not participate in any edge
    const edgeNodeSet = new Set<string>(edges.flatMap((e) => [e.source, e.target]));
    const filteredDemoNodes = allNodes.filter((n) => edgeNodeSet.has(n.id));
    return { nodes: filteredDemoNodes, edges };
  }, [graph, predictions]);

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

    // Zoom: ignore pointer events that originate on node circles so clicking
    // a node does not change the zoom scale/transform.
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // Start with a slightly zoomed-out view so the graph appears smaller initially
    try {
      svg.call(zoom.transform as any, d3.zoomIdentity.scale(0.8));
    } catch (e) {
      // ignore if the runtime disallows immediate transform
    }

    // nodeColors and uniqueTypes are computed outside so legend and rendering share them

    // Edge rendering
    const edgeLines = g.append("g").selectAll("line").data(edges).join("line").attr("stroke", "#cbd5e1").attr("stroke-width", 1.2);

    // Node rendering
      const circle = g
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => (d.group === "assignment" ? 8 : 12))
      .attr("fill", (d) => nodeColors[d.group] || "#64748b")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        // Store selection in a ref and update a small details state so
        // we do not re-run the main simulation effect (which depends on
        // nodes/edges only). Avoid calling setSelectedNode which caused
        // view resets.
        event.stopPropagation();
        selectedNodeRef.current = d as Node;
        setSelectedInfo({ id: d.id, group: d.group, risk: d.risk });
      });

    // Tooltip
    circle.append("title").text((d) => `${d.label || d.id}${d.risk !== undefined ? `\nRisk: ${(d.risk * 100).toFixed(0)}%` : ''}`);

    // Create an empty label group; labels will be managed by a separate effect
    g.append("g").attr("class", "labels");

    // D3 force simulation
    // Build a fast lookup map from id -> node object to avoid expensive array scans on each tick
    const nodesById = new Map<string, Node>(nodes.map((n) => [n.id, n]));
    nodesByIdRef.current = nodesById;
    gRef.current = g;

    const simulation = d3.forceSimulation<Node>(nodes);

    // Stronger repulsion and larger link distances help the layout spread
    simulation
      // Reduce repulsion so nodes sit more calmly
      .force("charge", d3.forceManyBody().strength(-20))
      // Smaller collision radius so nodes sit closer together
      .force("collide", d3.forceCollide().radius((d: any) => (d.group === "assignment" ? 6 : 12)))
      .force(
        "link",
        d3.forceLink(edges as any).id((d: any) => d.id).distance((d: any) => 100).strength(0.6)
      )
      // Use a single centering force rather than separate x/y pulls so nodes
      // can spread naturally while remaining roughly within view.
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Let the simulation breathe a bit longer for nicer spacing
    simulation.alphaTarget(0.12).alphaDecay(0.03);

    // Add dragging so users can pull nodes; dragging temporarily fixes nodes (fx/fy)
    const drag = d3
      .drag<SVGCircleElement, Node>()
      .on("start", (event: any, d: Node) => {
        if (event.sourceEvent) event.sourceEvent.stopPropagation();
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event: any, d: Node) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event: any, d: Node) => {
        if (!event.active) simulation.alphaTarget(0);
        // release fixed position so the node may settle under simulation
        d.fx = null as any;
        d.fy = null as any;
      });

    // Attach drag behavior to node circles so users can pull nodes
    try {
      (circle as any).call(drag as any);
    } catch (e) {
      // if attaching drag fails, ignore quietly
    }

    simulation.on("tick", () => {
      edgeLines
        .attr("x1", (d: any) => (typeof d.source === "object" ? d.source.x : nodesById.get(String(d.source))?.x) || 0)
        .attr("y1", (d: any) => (typeof d.source === "object" ? d.source.y : nodesById.get(String(d.source))?.y) || 0)
        .attr("x2", (d: any) => (typeof d.target === "object" ? d.target.x : nodesById.get(String(d.target))?.x) || 0)
        .attr("y2", (d: any) => (typeof d.target === "object" ? d.target.y : nodesById.get(String(d.target))?.y) || 0)
        .attr("stroke", "#cbd5e1")
        .attr("stroke-width", 1.2);

      circle.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);

      // Update any visible label position
      const labelSel = g.select("g.labels").selectAll("text");
      // Position labels using the canonical node positions from nodesById
      // (the bound datum for labels is a lightweight `selectedInfo` object,
      // so we must look up the live node coordinates instead of relying on
      // d.x/d.y from the datum).
      if (selectedNodeRef.current) {
        const live = nodesById.get(selectedNodeRef.current.id);
        if (live) {
          labelSel.attr("x", live.x!).attr("y", (live.y! - 14));
        }
      } else {
        // no selection -> ensure no labels are visible/positioned
        labelSel.attr("x", -9999).attr("y", -9999);
      }
    });

    return () => {
      simulation.stop();
      nodesByIdRef.current = null;
      gRef.current = null;
    };
  }, [nodes, edges, isFullscreen]);

  // Keep fullscreen state in sync with document; allow Esc to exit too
  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape" && document.fullscreenElement) {
        try {
          (document as any).exitFullscreen?.();
        } catch (e) {
          // ignore
        }
      }
    };
    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("fullscreenchange", onFsChange);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  // Update label text when selection changes without re-running the simulation
  useEffect(() => {
    const g = gRef.current;
    if (!g) return;
    const nodeMap = nodesByIdRef.current;

    const labelSel = g.select("g.labels").selectAll("text").data(selectedInfo ? [selectedInfo] : []);
    labelSel
      .join(
        (enter: any) =>
          enter
            .append("text")
            .attr("font-size", 12)
            .attr("fill", "#111827")
            .attr("font-weight", 600)
            .attr("text-anchor", "middle")
            .text((d: any) => d.id),
        (update: any) => update.text((d: any) => d.id),
        (exit: any) => exit.remove()
      )
      .attr("x", (d: any) => nodeMap?.get(d.id)?.x ?? 0)
      .attr("y", (d: any) => (nodeMap?.get(d.id)?.y ?? 0) - 14);
  }, [selectedInfo]);

  return (
    <div className={`flex flex-col md:flex-row gap-4 ${heightClass}`}>
      <div
        ref={wrapperRef}
        className="flex-1 bg-slate-50 border border-slate-100 rounded-lg overflow-hidden relative"
      >
        <svg ref={svgRef} className="w-full h-full block" onClick={() => setSelectedInfo(null)} />
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

          <div className="absolute top-2 right-2 z-40">
            {!isFullscreen ? (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  enterFullscreen();
                }}
                className="bg-white/90 p-2 rounded border border-slate-100 shadow-sm text-xs"
                aria-label="Enter fullscreen"
              >
                ⛶ Fullscreen
              </button>
            ) : (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  exitFullscreen();
                }}
                className="bg-white/90 p-2 rounded border border-slate-100 shadow-sm text-xs"
                aria-label="Exit fullscreen"
              >
                ✕ Exit
              </button>
            )}
          </div>
      </div>

      {selectedInfo && (
        <div className="w-full md:w-48 bg-white border border-slate-200 p-4 rounded-lg shadow-sm">
          <h3 className="text-xs font-bold text-slate-800 uppercase mb-2 border-b border-slate-100 pb-2">
            Details
          </h3>
          <div className="space-y-2 text-xs">
            <div>
              <span className="block text-slate-400 font-semibold">Node ID</span>
              <span className="font-mono text-slate-700">{selectedInfo.id}</span>
            </div>
            <div>
              <span className="block text-slate-400 font-semibold">Type</span>
              <span className="font-mono text-slate-700">{selectedInfo.group}</span>
            </div>
            {selectedInfo.risk !== undefined && (
              <div>
                <span className="block text-slate-400 font-semibold">Risk Score</span>
                <span
                  className={`font-bold ${
                    selectedInfo.risk > 0.7
                      ? "text-red-500"
                      : selectedInfo.risk > 0.4
                      ? "text-orange-400"
                      : "text-brand-green"
                  }`}
                >
                  {(selectedInfo.risk * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

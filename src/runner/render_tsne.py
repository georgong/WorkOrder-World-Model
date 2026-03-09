# render_tsne.py — Serve a t-SNE viewer with color-by attribute/neighbor feature.
# Usage: uvicorn src.runner.render_tsne:app --host 127.0.0.1 --port 8050
#        Or: python -m src.runner.render_tsne --json runs/tsne_weights/tsne_nodes.json --port 8050
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional FastAPI for server
try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    Query = None


# -------------------------
# Data prep: load JSON and build points + color options
# -------------------------
def load_tsne_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_color_options(nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """
    From nodes (each with x, y, id, attributes, neighbors, optional attr_list, x_features),
    build list of color options and a mapping option_id -> { type, values }.
    Returns (color_options_meta, option_id_to_values).
    color_options_meta: [{ "id": str, "label": str, "type": "categorical"|"continuous" }]
    option_values: id -> { type, values }.
    """
    color_options_meta: List[Dict[str, str]] = []
    option_values: Dict[str, Dict[str, Any]] = {}  # id -> { type, values }

    n = len(nodes)
    if n == 0:
        return color_options_meta, option_values

    # ----- Node label y (continuous) -----
    y_values: List[Optional[float]] = []
    has_y = False
    for node in nodes:
        attrs = node.get("attributes") or {}
        v = attrs.get("y")
        if v is not None:
            has_y = True
            y_values.append(float(v))
        else:
            y_values.append(None)
    if has_y:
        option_id = "attr_y"
        color_options_meta.append({"id": option_id, "label": "Node · y", "type": "continuous"})
        option_values[option_id] = {"type": "continuous", "values": y_values}

    # ----- Node attributes (categorical) -----
    attr_keys = ["engineer_id", "task_id", "task_type_id", "district_id", "department_id"]
    for key in attr_keys:
        values: List[Optional[int]] = []
        for node in nodes:
            attrs = node.get("attributes") or {}
            v = attrs.get(key)
            values.append(int(v) if v is not None else None)
        option_id = f"attr_{key}"
        color_options_meta.append({"id": option_id, "label": f"Node · {key}", "type": "categorical"})
        option_values[option_id] = {"type": "categorical", "values": values}

    # ----- Node's own features (attr_list + x_features) -----
    node_attr_list: Optional[List[str]] = None
    for node in nodes:
        al = node.get("attr_list")
        if isinstance(al, list) and len(al) > 0:
            node_attr_list = al
            break
    if node_attr_list:
        for attr_name in node_attr_list:
            try:
                idx = node_attr_list.index(attr_name)
            except ValueError:
                continue
            values = []
            for node in nodes:
                xf = node.get("x_features")
                if isinstance(xf, list) and 0 <= idx < len(xf):
                    values.append(float(xf[idx]))
                else:
                    values.append(None)
            option_id = f"node_feat_{attr_name}"
            color_options_meta.append({"id": option_id, "label": f"(node) · {attr_name}", "type": "continuous"})
            option_values[option_id] = {"type": "continuous", "values": values}

    # ----- Neighbors: (node_type, attr_name) -> for each node, first neighbor of that type, x[attr_idx]
    # Collect (node_type, attr_name) and attr_list index from first neighbor that has it
    neighbor_specs: Dict[Tuple[str, str], int] = {}  # (node_type, attr_name) -> attr_index
    for node in nodes:
        for nb in node.get("neighbors") or []:
            ntype = nb.get("node_type")
            al = nb.get("attr_list")
            if not ntype or not isinstance(al, list):
                continue
            for idx, an in enumerate(al):
                key = (ntype, an)
                if key not in neighbor_specs:
                    neighbor_specs[key] = idx
    for (ntype, attr_name), attr_idx in neighbor_specs.items():
        values = []
        for node in nodes:
            val = None
            for nb in node.get("neighbors") or []:
                if nb.get("node_type") != ntype:
                    continue
                x = nb.get("x")
                if isinstance(x, list) and 0 <= attr_idx < len(x):
                    val = float(x[attr_idx])
                break
            values.append(val)
        option_id = f"neighbor_{ntype}_{attr_name}"
        color_options_meta.append({"id": option_id, "label": f"{ntype} · {attr_name}", "type": "continuous"})
        option_values[option_id] = {"type": "continuous", "values": values}

    return color_options_meta, option_values


def build_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes = data.get("nodes") or []
    points = [{"x": n["x"], "y": n["y"], "id": n["id"]} for n in nodes]
    color_options_meta, option_values = build_color_options(nodes)
    return {
        "target_node_type": data.get("target_node_type", "assignments"),
        "points": points,
        "color_options": color_options_meta,
        "option_values": option_values,
    }


# -------------------------
# Server (FastAPI)
# -------------------------
if HAS_FASTAPI:
    app = FastAPI(title="t-SNE Weights Viewer")

    # Loaded on first request or at startup
    _DATA: Optional[Dict[str, Any]] = None
    _JSON_PATH: Optional[Path] = None

    def get_data(json_path: Optional[Path] = None) -> Dict[str, Any]:
        global _DATA, _JSON_PATH
        path = json_path or _JSON_PATH
        if path is None:
            path = Path("runs/tsne_weights/tsne_nodes.json")
        if _DATA is None or (_JSON_PATH is not None and path != _JSON_PATH):
            _JSON_PATH = path
            _DATA = load_tsne_json(path)
        return _DATA

    @app.get("/data", response_model=None)
    def api_data(json_path: Optional[str] = Query(None, alias="json")):
        path = Path(json_path).resolve() if json_path else None
        data = get_data(path)
        return build_payload(data)

    @app.get("/", response_class=HTMLResponse)
    def index(json_path: Optional[str] = Query(None, alias="json")):
        path = Path(json_path).resolve() if json_path else None
        data = get_data(path)
        payload = build_payload(data)
        # Embed payload and HTML so we don't need a separate static dir
        html = _render_html(payload)
        return HTMLResponse(html)

    def _render_html(initial_payload: Dict[str, Any]) -> str:
        payload_js = json.dumps(initial_payload)
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>t-SNE Weights Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 12px; background: #ffffff; color: #111827; }
    h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 12px 0; color: #111827; }
    .controls { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
    .controls label { font-size: 0.9rem; color: #374151; }
    .controls select { padding: 6px 10px; border-radius: 6px; border: 1px solid #d1d5db; background: #ffffff; color: #111827; min-width: 280px; }
    #legend { font-size: 0.8rem; color: #4b5563; margin-bottom: 8px; max-height: 120px; overflow-y: auto; }
    #legend span.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 4px; vertical-align: middle; }
    #plot { width: 100%; height: calc(100vh - 120px); min-height: 400px; }
  </style>
</head>
<body>
  <h1>t-SNE · Color by attribute / neighbor feature</h1>
  <div class="controls">
    <label for="colorBy">Color by:</label>
    <select id="colorBy"></select>
  </div>
  <div id="legend"></div>
  <div id="plot"></div>
  <script>
    const payload = """ + payload_js + """;
    const points = payload.points;
    const colorOptions = payload.color_options || [];
    const optionValues = payload.option_values || {};

    const select = document.getElementById('colorBy');
    const legendDiv = document.getElementById('legend');
    colorOptions.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt.id;
      o.textContent = opt.label;
      select.appendChild(o);
    });

    function getColorValues(optionId) {
      const o = optionValues[optionId];
      if (!o) return null;
      return o.values;
    }

    const layout = {
      margin: { t: 24, r: 24, b: 32, l: 32 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      xaxis: { title: 't-SNE 1', gridcolor: '#e5e7eb', zerolinecolor: '#9ca3af' },
      yaxis: { title: 't-SNE 2', gridcolor: '#e5e7eb', zerolinecolor: '#9ca3af' },
      font: { color: '#111827', size: 12 },
      showlegend: false
    };

    function updatePlot(optionId) {
      const vals = getColorValues(optionId);
      const opt = colorOptions.find(o => o.id === optionId);
      const isCategorical = opt && opt.type === 'categorical';

      const x = points.map(p => p.x);
      const y = points.map(p => p.y);
      const ids = points.map(p => p.id);

      let trace;
      if (vals && vals.length === points.length) {
        const valid = vals.map((v, i) => v != null && v !== '');
        const xv = x.filter((_, i) => valid[i]);
        const yv = y.filter((_, i) => valid[i]);
        const vv = vals.filter((_, i) => valid[i]);
        const idv = ids.filter((_, i) => valid[i]);

        if (isCategorical) {
          const uniq = [...new Set(vv)];
          const pal = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
            '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94','#f7b6d2','#c7c7c7','#dbdb8d','#9edae5'];
          const colorMap = {};
          uniq.forEach((u, i) => colorMap[u] = pal[i % pal.length]);
          const colors = vv.map(v => colorMap[v]);
          // Build simple legend for categorical options (e.g. departments)
          const maxLegendItems = 30;
          const shown = uniq.slice(0, maxLegendItems);
          if (legendDiv) {
            legendDiv.innerHTML = shown.map(v => {
              const c = colorMap[v];
              return `<div><span class="swatch" style="background:${c};"></span>${v}</div>`;
            }).join('');
            if (uniq.length > maxLegendItems) {
              legendDiv.innerHTML += `<div>… +${uniq.length - maxLegendItems} more</div>`;
            }
          }
          trace = {
            x: xv, y: yv, mode: 'markers', type: 'scatter',
            marker: { size: 4, opacity: 0.6, color: colors, line: { width: 0 } },
            text: idv.map(id => 'id: ' + id),
            hovertemplate: '%{text}<br>t-SNE: (%{x:.2f}, %{y:.2f})<extra></extra>'
          };
        } else {
          if (legendDiv) {
            legendDiv.innerHTML = '';
          }
          // Continuous: for y specifically, use a non-linear mapping on the color axis
          // but keep the colorbar tick labels in the original y space.
          let colorValues = vv;
          let cmin = null;
          let cmax = null;
          let colorbar = { title: opt ? opt.label : '' };
          if (opt && opt.id === 'attr_y') {
            const transformY = (v) => {
              if (v == null) return null;
              let u = v;
              if (u < 0) u = 0;
              if (u > 7) u = 7;
              // emphasize 0–1: map [0,1] -> [0,3], [1,7] -> [3,7]
              if (u <= 1.0) {
                return u * 3.0;
              }
              return 3.0 + (u - 1.0) * (4.0 / 6.0);
            };
            colorValues = vv.map(v => v == null ? null : transformY(v));
            cmin = 0;
            cmax = 7;
            // colorbar ticks: positions in transformed space, labels in original y
            const tickvals = [];
            const ticktext = [];
            for (let yTick = 0; yTick <= 7; yTick++) {
              tickvals.push(transformY(yTick));
              ticktext.push(String(yTick));
            }
            colorbar = {
              title: opt ? opt.label : '',
              tickmode: 'array',
              tickvals: tickvals,
              ticktext: ticktext,
            };
          }
          trace = {
            x: xv, y: yv, mode: 'markers', type: 'scatter',
            marker: {
              size: 4,
              opacity: 0.6,
              color: colorValues,
              colorscale: 'Viridis',
              cmin: cmin,
              cmax: cmax,
              colorbar: colorbar,
              line: { width: 0 }
            },
            text: idv.map((id, i) => 'id: ' + id + '<br>value: ' + vv[i]),
            hovertemplate: '%{text}<br>t-SNE: (%{x:.2f}, %{y:.2f})<extra></extra>'
          };
        }
      } else {
        trace = {
          x: x, y: y, mode: 'markers', type: 'scatter',
          marker: { size: 4, opacity: 0.6, color: '#71717a', line: { width: 0 } },
          text: ids.map(id => 'id: ' + id),
          hovertemplate: '%{text}<br>t-SNE: (%{x:.2f}, %{y:.2f})<extra></extra>'
        };
      }
      Plotly.react('plot', [trace], { ...layout }, { responsive: true });
    }

    select.addEventListener('change', () => updatePlot(select.value));
    // Initial color: prefer y if available
    let initialId = null;
    const yOpt = colorOptions.find(o => o.id === 'attr_y');
    if (yOpt) {
      initialId = yOpt.id;
    } else if (colorOptions.length) {
      initialId = colorOptions[0].id;
    }
    if (initialId) {
      select.value = initialId;
      updatePlot(initialId);
    }
  </script>
</body>
</html>"""
else:
    app = None


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Serve t-SNE viewer (color by node/neighbor attributes).")
    parser.add_argument("--json", type=str, default="runs/tsne_weights/tsne_nodes.json", help="Path to tsne_nodes.json")
    parser.add_argument("--port", type=int, default=8050, help="Port for the server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    args = parser.parse_args()

    if not HAS_FASTAPI:
        raise SystemExit("Install fastapi to run the server: pip install fastapi")

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Pre-load data into app
    global _DATA, _JSON_PATH
    if app is not None:
        _JSON_PATH = json_path.resolve()
        _DATA = load_tsne_json(_JSON_PATH)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

from typing import Dict, List
from collections import defaultdict
from pathlib import Path

class PipelineLogger:
    def __init__(self):
        self.logs: Dict[str, List[str]] = defaultdict(list)

    def log(self, pipeline: str, info: str) -> None:
        self.logs[pipeline].append(info)

    def dump(self, path: str | Path) -> None:
        path = Path(path)

        lines = []

        for pipeline, infos in self.logs.items():
            lines.append("=" * 14)
            lines.append(f"(pipeline) {pipeline} :")
            lines.append("=" * 14)

            for info in infos:
                lines.append(info)

            lines.append("=" * 14)
            lines.append("")

        content = "\n".join(lines)

        path.write_text(content, encoding="utf-8")
from predictor.config.pipeline_config import PipelineConfig, build_default_pipeline_config
from pathlib import Path

_SEP = "─" * 60
CFG: PipelineConfig = build_default_pipeline_config()
SAVE_DIR: Path = Path.cwd().parent.parent.parent

_C = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "home":       "#2f81f7",
    "home_pens":  "#6aabfc",
    "draw":       "#8b949e",
    "away":       "#f78166",
    "away_pens":  "#f7a898",
    "accent":     "#3fb950",
    "text":       "#e6edf3",
    "text_muted": "#8b949e",
}
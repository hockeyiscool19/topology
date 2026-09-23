"""Turn real-world terrain into layered CNC carving files (SVG pockets + STL)."""

from topology.config import BoardSpec, JobSpec, LayerSpec
from topology.pipeline import JobResult, run_job

__all__ = ["BoardSpec", "JobSpec", "LayerSpec", "JobResult", "run_job"]

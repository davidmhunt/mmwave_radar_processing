"""Dataset model wrapper for GUI access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from cpsl_datasets.cpsl_ds import CpslDS  # type: ignore
from mmwave_radar_processing.logging.logger import get_logger


class DatasetModel:
    """Thin wrapper around CpslDS for GUI use."""

    def __init__(
        self,
        params_path: Path,
        logger=None,
    ) -> None:
        """Initialize the dataset model from a YAML parameters file.

        Args:
            params_path: Path to the dataset parameters YAML.
            logger: Optional logger instance.
        """
        self.logger = logger or get_logger(__name__)
        self.dataset: Optional[CpslDS] = None
        self.params = self._load_params(params_path)
        self.load_from_params(self.params)

    def _load_params(self, params_path: Path) -> Dict[str, Any]:
        """Load dataset parameters from YAML.

        Args:
            params_path: Path to the YAML file.

        Returns:
            Parsed parameter dictionary.
        """
        params_path = Path(params_path)
        self.logger.info("Loading dataset params from %s", params_path)
        with params_path.open("r") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    def load_from_params(self, params: Dict[str, Any]) -> None:
        """Load a dataset using provided parameters.

        Args:
            params: Dataset parameter mapping (including CpslDS kwargs).
        """
        dataset_params = params.get("dataset", {})
        dataset_path = dataset_params.get("dataset_path", "")
        self.logger.info("Loading dataset from %s", dataset_path)
        self.dataset = CpslDS(**dataset_params)
        self.params = params

    def frame_count(self) -> int:
        """Return the number of frames available in the dataset.

        Returns:
            Frame count if dataset is loaded, otherwise 0.
        """
        if self.dataset is None:
            return 0
        return getattr(self.dataset, "num_frames", 0)

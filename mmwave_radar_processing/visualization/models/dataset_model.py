"""Dataset model wrapper for GUI access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from cpsl_datasets.cpsl_ds import CpslDS  # type: ignore
from mmwave_radar_processing.logging.logger import get_logger


class DatasetModel:
    """Thin wrapper around CpslDS for GUI use."""

    def __init__(
        self,
        params: Union[Path, Dict[str, Any]],
        logger=None,
        dataset_path_override: Optional[Path] = None,
    ) -> None:
        """Initialize the dataset model from a YAML parameters file or dict.

        Args:
            params: Path to the dataset parameters YAML or already-parsed dict.
            logger: Optional logger instance.
            dataset_path_override: Optional dataset path override.
        """
        self.logger = logger or get_logger(__name__)
        self.dataset: Optional[CpslDS] = None
        self.params = self._load_params(params)
        self.load_from_params(self.params, dataset_path_override=dataset_path_override)

    def _load_params(self, params: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load dataset parameters from YAML or dict.

        Args:
            params: Path to the YAML file or a dict of parameters.

        Returns:
            Parsed parameter dictionary.
        """
        if isinstance(params, dict):
            return params
        params_path = Path(params)
        self.logger.info("Loading dataset params from %s", params_path)
        with params_path.open("r") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    def load_from_params(
        self, params: Dict[str, Any], dataset_path_override: Optional[Path] = None
    ) -> None:
        """Load a dataset using provided parameters.

        Args:
            params: Dataset parameter mapping (including CpslDS kwargs).
            dataset_path_override: Optional dataset path override.
        """
        dataset_params = params.get("dataset", {}).copy()
        if dataset_path_override:
            dataset_params["dataset_path"] = str(dataset_path_override)
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

    def get_adc_data(self, frame_idx: int) -> Optional[Any]:
        """Get ADC data for a specific frame.

        Args:
            frame_idx: Index of the frame to retrieve.

        Returns:
            ADC data cube if available, otherwise None.
        """
        if self.dataset is None:
            return None
        return self.dataset.get_radar_adc_data(frame_idx)

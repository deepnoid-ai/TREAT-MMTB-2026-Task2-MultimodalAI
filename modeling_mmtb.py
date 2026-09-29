import csv
from pathlib import Path

import torch

from .configuration_mmtb import MMTBConfig
from .modeling_base import MMTBBaseModel
from .predict import TBModel, Task2Model, discover_images


class MMTBModel(MMTBBaseModel):
    config_class = MMTBConfig
    task = 2

    @staticmethod
    def checkpoint_indices(config):
        if config.mode != "cls":
            raise ValueError("task 2 supports mode='cls' only")
        return range(4)

    @classmethod
    def build_models(cls, config):
        return [TBModel(config.cls_spec) for _ in cls.checkpoint_indices(config)]

    @torch.inference_mode()
    def predict(self, input_path, output_dir=None, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        paths = discover_images(input_path)
        pipeline = Task2Model(self.models, self.config.cls_spec)
        results = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            results.extend({"filename": path.name, "label": label}
                           for path, label in zip(batch, pipeline.predict_batch(batch)))
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "prediction.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["filename", "TB/Normal"])
                writer.writerows((result["filename"], result["label"]) for result in results)
        return results

    def forward(self, input_path, output_dir=None, batch_size=32):
        return self.predict(input_path, output_dir=output_dir, batch_size=batch_size)

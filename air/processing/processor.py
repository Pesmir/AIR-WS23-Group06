import abc
import polars as pl
import os


class Processor:
    def __init__(self, name):
        self.name = name
        self._check_point_path = f"./air/data/checkpoints/{name}.ipc"

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        print(f"Processing {self.name}...")
        if os.path.exists(self._check_point_path):
            print(f"Found checkpoint for {self.name}, loading...")
            result = pl.read_ipc(self._check_point_path)
        else:
            result = self._process_inner(data)
            print(f"Saving checkpoint for {self.name} in {self._check_point_path}...")
            result.write_ipc(self._check_point_path)
        return result

    @abc.abstractmethod
    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        ...

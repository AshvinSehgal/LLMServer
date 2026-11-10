"""Reads a checkpoint's safetensors files one tensor at a time.

Loading a model through `from_pretrained` materializes every weight in memory first (16 GB for an 8B model in bf16). This reader
memory-maps the files and hands out single tensors, so the loader can quantize and move one decoder layer at a time.
"""
import json
import os

from safetensors import safe_open

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


class CheckpointReader:
    def __init__(self, directory):
        self.directory = directory
        index_path = os.path.join(directory, INDEX_NAME)
        if os.path.exists(index_path):
            with open(index_path) as handle:
                self.key_to_file = json.load(handle)["weight_map"]
        else:
            with safe_open(os.path.join(directory, SINGLE_NAME), "pt") as handle:
                self.key_to_file = {key: SINGLE_NAME for key in handle.keys()}
        self.handles = {}

    @classmethod
    def open(cls, model_name):
        """A reader for a local directory or a Hub repo (downloading the safetensors files), or None if there are none."""
        directory = model_name if os.path.isdir(model_name) else None
        if directory is None:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import EntryNotFoundError
            try:
                directory = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
            except EntryNotFoundError:
                return None
        has_weights = os.path.exists(os.path.join(directory, INDEX_NAME)) or os.path.exists(os.path.join(directory, SINGLE_NAME))
        return cls(directory) if has_weights else None

    def has(self, key):
        return key in self.key_to_file

    def get(self, key):
        file_name = self.key_to_file[key]
        if file_name not in self.handles:
            self.handles[file_name] = safe_open(os.path.join(self.directory, file_name), "pt")
        return self.handles[file_name].get_tensor(key)

    def with_prefix(self, prefix):
        """{name without prefix: tensor} for every key that starts with prefix."""
        return {key[len(prefix):]: self.get(key) for key in self.key_to_file if key.startswith(prefix)}

    def close(self):
        self.handles.clear()

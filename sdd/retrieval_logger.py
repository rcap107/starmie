import copy
import csv
import datetime as dt
import json
from pathlib import Path

class SimpleIndexLogger:
    def __init__(
        self,
        index_name,
        step,
        data_lake_version,
        index_parameters = None,
        query_parameters = None,
        log_path = "results/index_logging.txt",
    ) -> None:
        self.exp_name = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        self.log_path = log_path
        self.index_name = index_name
        self.data_lake_version = data_lake_version
        self.step = step
        self.index_parameters = index_parameters if index_parameters is not None else {}
        self.query_parameters = query_parameters if query_parameters is not None else {}

        self.timestamps = {}
        self.durations = {}

        self.memory_usage = {
            "create": None,
            "query": None,
            "inference": None,
            "peak_create": None,
            "peak_query": None,
            "peak_inference": None,
        }

        self.query_results = {"n_candidates": 0}

        self.header = [
            "data_lake_version",
            "index_name",
            "n_jobs",
            "base_table",
            "query_column",
            "step",
            "time_create",
            "time_save",
            "time_load",
            "time_query",
            "peak_create",
            "peak_query",
            "peak_inference",
            "n_candidates",
        ]

    def update_query_parameters(self, query_table, query_column):
        self.query_parameters["base_table"] = query_table
        self.query_parameters["query_column"] = query_column

    def start_time(self, label: str, cumulative: bool = False):
        """Wrapper around the `mark_time` function for better clarity.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.
        """
        return self.mark_time(label, cumulative)

    def end_time(self, label, cumulative = False):
        if label not in self.timestamps:
            raise KeyError(f"Label {label} was not found.")
        return self.mark_time(label, cumulative)

    def mark_time(self, label, cumulative = False):
        """Given a `label`, add a new timestamp if `label` isn't found, otherwise
        mark the end of the timestamp and add a new duration.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.

        """
        if label not in self.timestamps:
            self.timestamps[label] = [dt.datetime.now(), None]
            self.durations["time_" + label] = 0
        else:
            self.timestamps[label][1] = dt.datetime.now()
            this_segment = self.timestamps[label]
            if cumulative:
                self.durations["time_" + label] += (
                    this_segment[1] - this_segment[0]
                ).total_seconds()
            else:
                self.durations["time_" + label] = (
                    this_segment[1] - this_segment[0]
                ).total_seconds()

    def mark_memory(self, mem_usage, label):
        """Record the memory usage for a given section of the code.

        Args:
            mem_usage (list): List containing the memory usage and timestamps.
            label (str): One of "fit", "predict", "test".

        Raises:
            KeyError: Raise KeyError if the label is not correct.
        """
        if label in self.memory_usage:
            self.memory_usage[label] = mem_usage
            self.memory_usage[f"peak_{label}"] = max(
                _[0] for _ in self.memory_usage[label]
            )
        else:
            raise KeyError(f"Label {label} not found in mem_usage.")

    def to_dict(self):
        values = [
            self.data_lake_version,
            self.index_name,
            self.index_parameters.get("n_jobs", 1),
            self.query_parameters.get("base_table", ""),
            self.query_parameters.get("query_column", ""),
            self.step,
            self.durations.get("time_create", 0),
            self.durations.get("time_save", 0),
            self.durations.get("time_load", 0),
            self.durations.get("time_query", 0),
            self.memory_usage.get("peak_create", 0),
            self.memory_usage.get("peak_query", 0),
            self.memory_usage.get("peak_inference", 0),
            self.query_results.get("n_candidates", 0),
        ]

        return dict(zip(self.header, values))

    def to_logfile(self):
        if Path(self.log_path).exists():
            with open(self.log_path, "a") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.header)
                writer.writerow(self.to_dict())
        else:
            with open(self.log_path, "w") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.header)
                writer.writeheader()
                writer.writerow(self.to_dict())

    def write_to_json(self, root_path="results/profiling/"):
        res_dict = copy.deepcopy(vars(self))
        res_dict["index_parameters"] = {
            k: str(v) for k, v in res_dict["index_parameters"].items()
        }
        res_dict["timestamps"] = {
            k: v.isoformat()
            for k, v in res_dict["timestamps"].items()
            if isinstance(v, dt.datetime)
        }

        if Path(root_path).exists():
            dest_path = Path(root_path, self.exp_name + ".json")
            with open(dest_path, "w") as fp:
                json.dump(res_dict, fp, indent=2)
        else:
            raise IOError(f"Invalid path {root_path}")

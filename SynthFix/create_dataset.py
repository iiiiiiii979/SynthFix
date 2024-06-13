import json
from typing import Any, List
from typing import Tuple
from typing import Dict

JsonDict = Dict[str, Any]


class Instruction:
    def __init__(
        self,
        inst_type: str,
        text: str,
        line_number: str,
        line_column: str,
        global_idx: str,
        description: str,
        relativ_pos: str,
    ):
        self.type = inst_type
        self.text = text
        self.line_number = line_number
        self.line_column = line_column
        self.global_idx = global_idx
        self.description = description
        self.relativ_pos = relativ_pos

    def GetDescription(self) -> str:
        return self.description


class LinterReport:
    def __init__(
        self,
        rule_id: str,
        message: str,
        evidence: str,
        col_begin: str,
        col_end: str,
        line_begin: str,
        line_end: str,
        severity: str,
    ):
        self.rule_id = rule_id
        self.message = message
        self.evidence = evidence
        self.col_begin = col_begin
        self.col_end = col_end
        self.line_begin = line_begin
        self.line_end = line_end
        self.severity = severity


class DataPoint:
    def __init__(
        self,
        source_code: str,
        target_code: str,
        warning_line: str,
        linter_report: LinterReport,
        instructions: List[Instruction],
        source_file: str,
        target_file: str,
        repo: str,
        source_filename: str,
        target_filename: str,
        source_changeid: str,
        target_changeid: str,
    ):

        self.source_code = source_file
        self.target_code = target_file
        self.warning_line = warning_line
        self.linter_report = linter_report
        self.instructions = instructions
        self.source_file = source_file
        self.target_file = target_file
        self.repo = repo
        self.source_filename = source_filename
        self.target_filename = target_filename
        self.source_changeid = source_changeid
        self.target_changeid = target_changeid

    def GetDescription(self) -> str:
        desc = (
            "WARNING\n"
            + self.linter_report.rule_id
            + " "
            + self.linter_report.message
            + " at line: "
            + str(self.linter_report.line_begin)
            + "\n"
        )

        desc += "WARNING LINE\n" + self.warning_line + "\n"
        desc += "SOURCE PATCH\n" + self.source_code + "\nTARGET PATCH\n" + self.target_code + "\n"

        desc += "INSTRUCTIONS\n"
        for inst in self.instructions:
            desc += inst.GetDescription() + "\n"
        return desc

    def GetT5Representation(self, include_warning: bool) -> Tuple[str, str]:
        if include_warning:
            inputs = (
                "fix "
                + self.linter_report.rule_id
                + " "
                + self.linter_report.message
                + " "
                + self.warning_line
                + ":\n"
                + self.source_code
                + " </s>"
            )
        else:
            inputs = "fix " + self.source_code + " </s>"
        outputs = self.target_code + " </s>"
        return inputs, outputs


def GetDataAsPython(data_json_path: str) -> List[DataPoint]:
    with open(data_json_path, "r", errors="ignore") as f:
        data_json = json.load(f)

    # converts a data point in json format to a data point in python object
    def FromJsonToPython(sample: JsonDict) -> DataPoint:
        linter_report = LinterReport(
            sample["linter_report"]["rule_id"],
            sample["linter_report"]["message"],
            sample["linter_report"]["evidence"],
            sample["linter_report"]["col_begin"],
            sample["linter_report"]["col_end"],
            sample["linter_report"]["line_begin"],
            sample["linter_report"]["line_end"],
            sample["linter_report"]["severity"],
        )

        instructions = []
        for inst in sample["instructions"]:
            instruction = Instruction(
                inst["type"],
                inst["text"],
                inst["line_number"],
                inst["line_column"],
                inst["global_idx"],
                inst["description"],
                inst["relativ_pos"],
            )
            instructions.append(instruction)

        data_point = DataPoint(
            sample["source_code"],
            sample["target_code"],
            sample["warning_line"],
            linter_report,
            instructions,
            sample["source_file"],
            sample["target_file"],
            sample["repo"],
            sample["source_filename"],
            sample["target_filename"],
            sample["source_changeid"],
            sample["target_changeid"],
        )

        return data_point

    data = [FromJsonToPython(sample) for sample in data_json]
    return data

from collections import defaultdict
from typing import Any, DefaultDict, List, Dict

from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.model_selection import train_test_split
import torch
from transformers import BatchEncoding



def extract_warning_types(data: List[DataPoint]) -> List[str]:
    all_warnings: List[str] = []
    for sample in data:
        if sample.linter_report.rule_id not in all_warnings:
            all_warnings.append(sample.linter_report.rule_id)
    return all_warnings


def filter_rule(data: List[DataPoint], rule_type: str) -> List[DataPoint]:
    filtered_data: List[DataPoint] = []
    for point in data:
        if point.linter_report.rule_id == rule_type:
            filtered_data.append(point)
    return filtered_data


def split_filtered(filtered_data: List[DataPoint], include_warning: bool, model_name: str, seed=13):
    filtered_data_temp = filtered_data

    inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
    outputs = [
        data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp
    ]

    test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, outputs, shuffle=True, random_state=seed, test_size=test_size
    )

    train_info, test_info = train_test_split(
        filtered_data, shuffle=True, random_state=seed, test_size=test_size
    )

    val_size = 0.1 if len(train_inputs) >= 10 else 1 / len(train_inputs)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size
    )

    train_info, val_info = train_test_split(
        train_info, shuffle=True, random_state=seed, test_size=test_size
    )

    return (
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
        train_info,
        val_info,
        test_info,
    )


def create_data(
    data: List[DataPoint], linter_warnings: List[str], include_warning: bool, model_name: str
):
    train: List[str] = []
    train_labels: List[str] = []
    val: List[str] = []
    val_labels: List[str] = []

    test: DefaultDict[str, List[str]] = defaultdict(list)
    test_labels: DefaultDict[str, List[str]] = defaultdict(list)
    n_test_samples = 0

    train_info: List[DataPoint] = []
    val_info: List[DataPoint] = []
    test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)

    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)
        (
            train_w,
            train_w_labels,
            val_w,
            val_w_labels,
            test_w,
            test_w_labels,
            train_w_info,
            val_w_info,
            test_w_info,
        ) = split_filtered(filtered_data, include_warning, model_name)

        train += train_w
        train_labels += train_w_labels

        val += val_w
        val_labels += val_w_labels

        train_info += train_w_info
        val_info += val_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels

        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print(
        "train size: {}\nval size: {}\ntest size: {}".format(len(train), len(val), n_test_samples)
    )
    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info


class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, targets: BatchEncoding):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.target_encodings["input_ids"][index], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def create_dataset(
    inputs: List[str],
    labels: List[str],
    tokenizer: PreTrainedTokenizer,
    pad_truncate: bool,
    max_length=None,
) -> BugFixDataset:
    if max_length is not None:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
    else:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset

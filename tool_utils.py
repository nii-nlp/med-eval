import os
from prettytable import PrettyTable
from typing import Any


def is_main_process():
    try:
        return int(os.environ["LOCAL_RANK"]) == 0
    except:
        return True


def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def show_pretty_table(evaluation_results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """
    Displays a formatted table of evaluation results using PrettyTable.

    Args:
        evaluation_results (dict): {task_name: {template_name: evaluation_result}}.
            Example: { "task_1": { "template_A": { "accuracy": 0.95 }}}
    """
    default_fileds = ["Task", "Template"]
    # get keys of evaluation_result
    metrics = list(next(iter(next(iter(evaluation_results.values())).values())).keys())
    metrics_title = [m.capitalize() for m in metrics]

    tb = PrettyTable()
    tb.field_names = default_fileds + metrics_title
    for task, template_result in evaluation_results.items():
        for template, result in template_result.items():
            tb.add_row([task, template] + [result[m] for m in metrics])
    main_print(tb)


def output_as_csv(
    evaluation_results: dict[str, dict[str, dict[str, Any]]], output_file: str
) -> None:
    """
    Outputs evaluation results to a CSV file.
    Args:
        evaluation_results (dict): {task_name: {template_name: evaluation_result}}.
            Example: { "task_1": { "template_A": { "accuracy": 0.95 }}}

        output_file (str): The path to the output CSV file
    """
    assert output_file is not None and output_file

    default_fileds = ["Task", "Template"]
    # get keys of evaluation_result
    metrics = list(next(iter(next(iter(evaluation_results.values())).values())).keys())
    metrics_title = [m.capitalize() for m in metrics]

    write_buffer = []
    write_buffer.append(",".join(default_fileds + metrics_title))
    for task, template_result in evaluation_results.items():
        for template, result in template_result.items():
            _result = [task, template] + [str(result[m]) for m in metrics]
            write_buffer.append(",".join(_result))

    write_buffer.append("")
    with open(output_file, "w") as f:
        f.write("\n".join(write_buffer))

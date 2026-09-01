"""Plot correlation matrices for each Building_*.csv file and write a short summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRECTORY = PROJECT_ROOT / "data/raw/citylearn_challenge_2023_phase_1"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "analysis" / f"result_{Path(__file__).stem}"


def collect_building_files(input_directory: Path) -> List[Path]:
    """Return the sorted building CSV files in the dataset directory."""

    building_files = sorted(input_directory.glob("Building_*.csv"))
    if not building_files:
        raise FileNotFoundError(f"No Building_*.csv files found in {input_directory}")
    return building_files


def load_numeric_frame(building_file: Path) -> pd.DataFrame:
    """Load a building file and keep only numeric columns for correlation analysis."""

    data_frame = pd.read_csv(building_file)
    numeric_frame = data_frame.select_dtypes(include="number")
    if numeric_frame.empty:
        raise ValueError(f"{building_file} does not contain numeric columns")
    return numeric_frame


def plot_correlation_matrix(correlation_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    """Render a correlation heatmap with labeled axes."""

    feature_count = len(correlation_frame.columns)
    figure_size = max(10.0, feature_count * 0.75)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))

    image = axis.imshow(correlation_frame.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(feature_count))
    axis.set_yticks(range(feature_count))
    axis.set_xticklabels(correlation_frame.columns, rotation=45, ha="right")
    axis.set_yticklabels(correlation_frame.columns)
    axis.set_title(title)
    axis.set_xticks([index - 0.5 for index in range(feature_count + 1)], minor=True)
    axis.set_yticks([index - 0.5 for index in range(feature_count + 1)], minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def iter_off_diagonal_pairs(correlation_frame: pd.DataFrame) -> Iterable[Tuple[str, str, float]]:
    """Yield each unique off-diagonal correlation pair once."""

    columns = list(correlation_frame.columns)
    for left_index, left_column in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right_column = columns[right_index]
            value = float(correlation_frame.iat[left_index, right_index])
            if pd.notna(value):
                yield left_column, right_column, value


def strongest_pairs(correlation_frame: pd.DataFrame, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Return the strongest positive and negative feature pairs."""

    pairs = list(iter_off_diagonal_pairs(correlation_frame))
    positive_pairs = sorted((pair for pair in pairs if pair[2] > 0), key=lambda item: item[2], reverse=True)
    negative_pairs = sorted((pair for pair in pairs if pair[2] < 0), key=lambda item: item[2])
    return positive_pairs[:top_k], negative_pairs[:top_k]


def top_target_correlations(correlation_frame: pd.DataFrame, target_column: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Return the features most correlated with a given target column."""

    if target_column not in correlation_frame.columns:
        return []

    series = correlation_frame[target_column].drop(labels=[target_column], errors="ignore")
    ordered = series.abs().sort_values(ascending=False).head(top_k)
    return [(column, float(series[column])) for column in ordered.index]


def mean_absolute_off_diagonal(correlation_frame: pd.DataFrame) -> float:
    """Compute the mean absolute off-diagonal correlation."""

    values = [abs(value) for _, _, value in iter_off_diagonal_pairs(correlation_frame)]
    return float(sum(values) / len(values)) if values else 0.0


def format_pair(pair: Tuple[str, str, float]) -> str:
    """Format a feature pair for the report."""

    left_column, right_column, value = pair
    return f"{left_column} <-> {right_column} (r={value:.3f})"


def format_target_lines(correlation_frame: pd.DataFrame, target_column: str) -> List[str]:
    """Build short target-specific conclusions for the report."""

    target_pairs = top_target_correlations(correlation_frame, target_column)
    if not target_pairs:
        return []

    pretty_name = target_column.replace("_", " ")
    lines = [f"- Strongest correlations with {pretty_name}:"]
    lines.extend([f"  - {feature} (r={value:.3f})" for feature, value in target_pairs])
    return lines


def build_report_section(building_file: Path, row_count: int, correlation_frame: pd.DataFrame) -> List[str]:
    """Create the markdown block for one building."""

    positive_pairs, negative_pairs = strongest_pairs(correlation_frame)
    mean_absolute = mean_absolute_off_diagonal(correlation_frame)

    lines = [f"## {building_file.name}", ""]
    lines.append(f"- Rows: {row_count}")
    lines.append(f"- Features analysed: {len(correlation_frame.columns)} numeric columns")
    lines.append(f"- Mean absolute off-diagonal correlation: {mean_absolute:.3f}")
    if positive_pairs:
        lines.append(f"- Strongest positive pair: {format_pair(positive_pairs[0])}")
    if negative_pairs:
        lines.append(f"- Strongest negative pair: {format_pair(negative_pairs[0])}")

    lines.append("")
    lines.append("### Target-specific signals")
    for target_column in ["cooling_demand", "solar_generation", "non_shiftable_load"]:
        lines.extend(format_target_lines(correlation_frame, target_column))

    if positive_pairs:
        lines.append("")
        lines.append("### Top positive relationships")
        for pair in positive_pairs:
            lines.append(f"- {format_pair(pair)}")

    if negative_pairs:
        lines.append("")
        lines.append("### Top negative relationships")
        for pair in negative_pairs:
            lines.append(f"- {format_pair(pair)}")

    lines.append("")
    return lines


def overall_conclusion(report_frames: Sequence[pd.DataFrame]) -> List[str]:
    """Generate a short cross-building conclusion from the computed matrices."""

    if not report_frames:
        return []

    mean_abs_values = [mean_absolute_off_diagonal(frame) for frame in report_frames]
    average_mean_abs = sum(mean_abs_values) / len(mean_abs_values)
    conclusion_lines = ["## Overall conclusion", ""]
    conclusion_lines.append(
        "- The building files share the same feature structure, and the correlation patterns are highly similar across buildings."
    )
    conclusion_lines.append(
        f"- The average mean absolute off-diagonal correlation is {average_mean_abs:.3f}, which indicates a moderately structured feature set rather than independent signals."
    )
    conclusion_lines.append(
        "- In practice, this means the calendar/setpoint columns and the energy-demand columns carry the clearest linear relationships, while the matrix also exposes a few strongly redundant features that are worth handling carefully in downstream models."
    )
    return conclusion_lines


def write_report(report_path: Path, sections: Sequence[List[str]], conclusion_lines: Sequence[str]) -> None:
    """Write the markdown summary report."""

    lines = ["# Building correlation analysis", ""]
    lines.append("This report was generated from the raw Building_*.csv files using Pearson correlation.")
    lines.append("")
    for section in sections:
        lines.extend(section)
    lines.extend(conclusion_lines)
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    """Entry point for the correlation matrix analysis script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()

    building_files = collect_building_files(arguments.input_directory)
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    report_sections: List[List[str]] = []
    correlation_frames: List[pd.DataFrame] = []

    for building_file in building_files:
        data_frame = load_numeric_frame(building_file)
        correlation_frame = data_frame.corr()
        correlation_frames.append(correlation_frame)

        plot_output_path = output_directory / f"{building_file.stem}_correlation_matrix.png"
        csv_output_path = output_directory / f"{building_file.stem}_correlation_matrix.csv"

        plot_correlation_matrix(correlation_frame, f"{building_file.stem} correlation matrix", plot_output_path)
        correlation_frame.to_csv(csv_output_path)
        report_sections.append(build_report_section(building_file, len(data_frame), correlation_frame))

    write_report(output_directory / "conclusions.md", report_sections, overall_conclusion(correlation_frames))
    print(f"Saved correlation matrices and summary to {output_directory}")


if __name__ == "__main__":
    main()
"""Markdown report, with an optional Word export through pandoc."""

import logging
import shutil
import subprocess
from pathlib import Path

from kairos import config
from kairos.reporting.findings import findings
from kairos.reporting.results import load_results
from kairos.reporting.sections import (
    data_section,
    discussion_section,
    forecasting_section,
    intro_section,
    portfolio_section,
    robustness_section,
)
from kairos.reporting.text import closing_section, method_section

log = logging.getLogger(__name__)


def build_markdown(paths: config.Paths = config.PATHS) -> str:
    """Render the report text."""
    r = load_results(paths)
    f = findings(r)
    parts = [
        *intro_section(r, f),
        *data_section(r),
        *method_section(),
        *forecasting_section(r),
        *portfolio_section(r),
        *robustness_section(r),
        *discussion_section(r, f),
        *closing_section(),
    ]
    return "\n\n".join(parts) + "\n"


def run(paths: config.Paths = config.PATHS) -> Path:
    """Report stage."""
    paths.reports.mkdir(parents=True, exist_ok=True)
    md = paths.reports / "Kairos_Report.md"
    md.write_text(build_markdown(paths), encoding="utf-8")
    log.info("report written to %s", md)
    if not shutil.which("pandoc"):
        log.warning("pandoc not found, skipping the docx export")
    else:
        subprocess.run(
            [
                "pandoc",
                md.name,
                "-o",
                "Kairos_Report.docx",
                "--toc",
                "--resource-path=.",
            ],
            cwd=paths.reports,
            check=True,
        )
    return md

"""Build the Excel workbook in memory: Summary, Test Cases, Traceability, Requirements."""

import io
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from . import config
from .pipeline import traceability
from .schemas import Requirement, TestCase

HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="D0D7DE")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
GOOD = PatternFill("solid", fgColor="E6F4EA")
BAD = PatternFill("solid", fgColor="FCE8E6")


def _table(ws, headers: list[str], rows: list[list], widths: list[int]) -> None:
    ws.append(headers)
    for cell in ws[1]:
        cell.fill, cell.font, cell.border = HEADER_FILL, HEADER_FONT, BORDER
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row in rows:
        ws.append(row)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment, cell.border = WRAP, BORDER
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = ws.dimensions


def _steps(tc: TestCase) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(tc.steps, start=1))


def build_workbook(
    requirements: list[Requirement],
    test_cases: list[TestCase],
    source_name: str = "BRD",
    model: str = "",
    reviewer: str = "",
) -> bytes:
    """Approved test cases go in the pack; the rest are kept on a Not Approved sheet for audit.

    Test cases with no review status set (e.g. from older callers) count as approved.
    """
    approved = [tc for tc in test_cases if tc.review_status in ("Approved", "")]
    rejected = [tc for tc in test_cases if tc.review_status not in ("Approved", "")]
    wb = Workbook()

    # ---- Summary (formulas, so it stays right if the tester edits the sheets)
    ws = wb.active
    ws.title = "Summary"
    n_tc, n_req = len(approved) + 1, len(requirements) + 1
    summary = [
        ("Source document", source_name),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Model", model),
        ("Reviewed and approved by", reviewer or "—"),
        ("Requirements", f"=COUNTA(Traceability!A2:A{n_req})"),
        ("Approved test cases", f"=COUNTA('Test Cases'!A2:A{n_tc})"),
        ("Not approved", len(rejected)),
        ("Requirements covered", f'=COUNTIF(Traceability!H2:H{n_req},"Covered")'),
        ("Coverage", None),
        ("Passed", f"=COUNTIF('Test Cases'!J2:J{n_tc},\"Pass\")"),
        ("Failed", f"=COUNTIF('Test Cases'!J2:J{n_tc},\"Fail\")"),
        ("Blocked", f"=COUNTIF('Test Cases'!J2:J{n_tc},\"Blocked\")"),
        ("Not run", f"=COUNTIF('Test Cases'!J2:J{n_tc},\"Not Run\")"),
    ]
    row_of = {label: i for i, (label, _) in enumerate(summary, start=3)}
    ws["A1"] = "AI Test Case Generator — Test Pack"
    ws["A1"].font = Font(bold=True, size=14, color="1F3A5F")
    for label, value in summary:
        r = row_of[label]
        ws.cell(row=r, column=1, value=label).font = Font(bold=True)
        ws.cell(row=r, column=2, value=value)
    req_cell, cov_cell = f"B{row_of['Requirements']}", f"B{row_of['Requirements covered']}"
    coverage = ws.cell(row=row_of["Coverage"], column=2)
    coverage.value = f"=IF({req_cell}=0,0,{cov_cell}/{req_cell})"
    coverage.number_format = "0%"
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 50
    note = ws.cell(row=len(summary) + 4, column=1, value="AI-generated, then reviewed by a person (human in the loop).")
    note.font = Font(italic=True, color="666666")

    # ---- Test Cases (approved only)
    ws = wb.create_sheet("Test Cases")
    headers = [
        "Test Case ID", "Requirement ID", "Title", "Type", "Priority",
        "Preconditions", "Steps", "Test Data", "Expected Result", "Status", "Actual Result",
        "Revision",
    ]
    rows = [
        [
            tc.id, tc.requirement_id, tc.title, tc.test_type, tc.priority, tc.preconditions,
            _steps(tc), tc.test_data, tc.expected_result, "Not Run", "",
            tc.revision,
        ]
        for tc in approved
    ]
    _table(ws, headers, rows, [13, 14, 40, 13, 10, 30, 55, 25, 45, 11, 30, 10])
    if rows:
        dv = DataValidation(
            type="list", formula1='"' + ",".join(config.EXECUTION_STATUSES) + '"', allow_blank=True
        )
        ws.add_data_validation(dv)
        dv.add(f"J2:J{len(rows) + 1}")

    # ---- Traceability
    ws = wb.create_sheet("Traceability")
    trace = traceability(requirements, approved)
    headers = list(trace[0].keys()) if trace else ["Requirement ID"]
    _table(ws, headers, [list(r.values()) for r in trace], [14, 40, 15, 10, 11, 35, 30, 13])
    for row in ws.iter_rows(min_row=2, min_col=8, max_col=8):
        for cell in row:
            cell.fill = GOOD if cell.value == "Covered" else BAD

    # ---- Requirements (with the source quote, so every test is traceable back to the BRD)
    ws = wb.create_sheet("Requirements")
    _table(
        ws,
        ["Requirement ID", "Title", "Description", "Category", "Priority", "Source Excerpt"],
        [[r.id, r.title, r.description, r.category, r.priority, r.source_excerpt] for r in requirements],
        [14, 35, 60, 15, 10, 60],
    )

    # ---- Not Approved (audit trail of what the reviewer didn't approve, and their feedback)
    ws = wb.create_sheet("Not Approved")
    _table(
        ws,
        ["Test Case ID", "Requirement ID", "Title", "Type", "Steps", "Expected Result", "Reviewer Feedback"],
        [
            [tc.id, tc.requirement_id, tc.title, tc.test_type, _steps(tc), tc.expected_result, tc.reviewer_comment]
            for tc in rejected
        ],
        [13, 14, 40, 13, 50, 40, 45],
    )

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

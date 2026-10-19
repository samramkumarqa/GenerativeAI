"""Generate samples/sample_brd.pdf, a short fictional BRD used by the demo and tests.

Run:  python scripts/make_sample_brd.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

OUT = Path(__file__).resolve().parent.parent / "samples" / "sample_brd.pdf"

TITLE = "Business Requirements Document: Checkout & Payments"
META = "Product: Leafcart Grocery App (fictional) · Version 1.2 · Status: Approved for build"

SECTIONS = [
    ("1. Purpose", [
        "This document describes the business requirements for the redesigned checkout "
        "journey of the Leafcart grocery app on web and mobile. It is the baseline for "
        "design, development and testing.",
    ]),
    ("2. Scope", [
        "In scope: cart review, delivery slot selection, promo codes, payment, order "
        "confirmation and order cancellation. Out of scope: product search, loyalty points "
        "and the warehouse picking system.",
    ]),
    ("3. Functional Requirements", None),
    ("4. Non-Functional Requirements", None),
    ("5. Assumptions", [
        "The payment gateway provides a sandbox environment for testing. Delivery slots are "
        "supplied by the existing Slot Service API. All prices are in INR and include GST.",
    ]),
]

FUNCTIONAL = [
    ("FR-01", "Cart minimum", "The customer cannot proceed to checkout if the cart subtotal is "
     "below INR 199. The Checkout button is disabled and the message 'Add items worth INR X more "
     "to checkout' is shown, where X is the shortfall.", "High"),
    ("FR-02", "Cart quantity limits", "Each product line allows a quantity from 1 to 10. "
     "Setting quantity to 0 removes the line after confirmation.", "Medium"),
    ("FR-03", "Delivery slot selection", "The customer must choose a delivery slot before "
     "payment. Slots are shown for the next 3 days in 2-hour windows between 07:00 and 21:00. "
     "Slots that are full or start within 90 minutes are greyed out and cannot be selected.", "High"),
    ("FR-04", "Delivery fee", "Orders with a subtotal of INR 499 or more get free delivery. "
     "Orders below INR 499 are charged a delivery fee of INR 35.", "High"),
    ("FR-05", "Promo codes", "The customer can apply one promo code per order. Codes are 6 to 12 "
     "alphanumeric characters and are not case sensitive. An invalid or expired code shows "
     "'This code is not valid'. The discount cannot exceed 50% of the subtotal.", "Medium"),
    ("FR-06", "Payment methods", "Supported payment methods are UPI, credit/debit card and "
     "cash on delivery. Cash on delivery is only available for orders up to INR 2,000.", "High"),
    ("FR-07", "Payment failure", "If a payment fails, the order is not placed, the cart is kept "
     "unchanged and the customer can retry. After 3 failed attempts within 15 minutes, card "
     "payment is blocked for that order for 30 minutes.", "High"),
    ("FR-08", "Order confirmation", "On successful payment the customer sees an order number in "
     "the format LC-YYYYMMDD-NNNNN and receives an SMS and email confirmation within 2 minutes.", "High"),
    ("FR-09", "Order cancellation", "The customer can cancel an order free of charge until 60 "
     "minutes before the start of the delivery slot. Refunds for prepaid orders are initiated "
     "to the original payment method within 24 hours.", "Medium"),
]

NON_FUNCTIONAL = [
    ("NFR-01", "Performance", "The checkout page loads within 2 seconds at the 95th percentile "
     "with up to 5,000 concurrent users.", "High"),
    ("NFR-02", "Security", "Card details are never stored by Leafcart; card entry uses the "
     "gateway's hosted fields. All checkout traffic uses TLS 1.2 or higher.", "High"),
    ("NFR-03", "Session timeout", "An idle checkout session expires after 15 minutes. The cart is "
     "preserved, but the selected slot is released.", "Medium"),
    ("NFR-04", "Accessibility", "All checkout screens meet WCAG 2.1 level AA, including keyboard "
     "navigation and screen-reader labels for every form field.", "Medium"),
]


def req_table(rows, styles):
    body = styles["BodyText"]
    data = [["ID", "Requirement", "Description", "Priority"]] + [
        [rid, Paragraph(name, body), Paragraph(desc, body), prio] for rid, name, desc, prio in rows
    ]
    t = Table(data, colWidths=[1.6 * cm, 3.2 * cm, 10.2 * cm, 1.8 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C0C7CF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    return t


def main():
    OUT.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(TITLE, styles["Title"]), Paragraph(META, styles["Italic"]), Spacer(1, 12)]
    for heading, paras in SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        if heading.startswith("3."):
            story.append(req_table(FUNCTIONAL, styles))
        elif heading.startswith("4."):
            story.append(req_table(NON_FUNCTIONAL, styles))
        else:
            story += [Paragraph(p, styles["BodyText"]) for p in paras]
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(OUT), pagesize=A4, title=TITLE, author="Sample").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

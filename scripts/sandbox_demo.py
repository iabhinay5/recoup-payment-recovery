"""End-to-end demo against the real Razorpay sandbox.

Run:  python scripts/sandbox_demo.py

Creates a real order in Razorpay test mode, opens a real Razorpay Checkout in your
browser, waits for you to fail the payment on purpose, then pulls the failure back through
Recoup and prints the decision chain: what Razorpay said, how the taxonomy classified it,
what the policy chose, and what the guardrails allowed.

Uses polling rather than webhooks on purpose. Webhooks need a public tunnel, which is one
more thing to fail on camera; the same event is available from the API. The webhook path
exists in ``recoup.gateway.server`` and is what a deployment would use.

No real money moves. The client refuses any key that is not rzp_test_.
"""

from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

from recoup.gateway import RazorpayClient, event_from_entity
from recoup.trace import explain

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
OFF = "\033[0m"

AMOUNT_PAISE = 125_000
CHECKOUT_PATH = Path("data/cache/checkout.html")
POLL_SECONDS = 3
POLL_LIMIT = 100

CHECKOUT_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Recoup sandbox</title>
<style>
 body {{ font-family: system-ui, sans-serif; background:#0d1117; color:#e6edf3;
        display:grid; place-items:center; height:100vh; margin:0; text-align:center }}
 .card {{ max-width: 460px; padding: 32px }}
 h1 {{ font-size:20px; margin:0 0 8px }}
 p {{ color:#8b949e; line-height:1.6; font-size:14px }}
 code {{ background:#161b22; padding:2px 6px; border-radius:4px; color:#7ee787 }}
 ol {{ text-align:left; color:#8b949e; font-size:13.5px; line-height:1.9; padding-left:22px }}
 li {{ margin-bottom:6px }}
 button {{ margin-top:20px; padding:12px 28px; font-size:15px; border:0; border-radius:6px;
          background:#2f81f7; color:#fff; cursor:pointer }}
</style>
<div class="card">
  <h1>Recoup &mdash; sandbox payment</h1>
  <p>Test mode. No real money moves.</p>
  <p><b>Make this fail on purpose.</b> Whichever method your account offers:</p>
  <ol>
    <li><b>Netbanking</b> &mdash; pick any bank, then click
        <code>Failure</code> on the mock bank page. Needs no credentials.</li>
    <li><b>Card</b> &mdash; any test card, any future expiry, any CVV, then enter an
        OTP of <b>fewer than 4 digits</b> to fail authentication.</li>
    <li><b>UPI</b> &mdash; enter <code>failure@razorpay</code> as the UPI ID.</li>
  </ol>
  <button id="pay">Pay &#8377;{amount}</button>
</div>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
 var opts = {{
   key: "{key_id}",
   order_id: "{order_id}",
   name: "Recoup Sandbox",
   description: "Buildathon demo - test mode",
   theme: {{ color: "#0E8C74" }},
   handler: function (r) {{
     document.querySelector(".card").innerHTML =
       "<h1>Payment succeeded</h1><p>Try again and use <code>failure@razorpay</code> "
     + "to produce the failure this demo is about.</p>";
   }},
   modal: {{ ondismiss: function () {{}} }}
 }};
 document.getElementById("pay").onclick = function () {{ new Razorpay(opts).open(); }};
</script>
"""


def show(label: str, value: str, colour: str = "") -> None:
    print(f"  {label:<22} {colour}{value}{OFF}")


def describe_failure(payment: dict[str, Any]) -> None:
    """Print the decision chain for one real failed payment.

    The chain comes from ``recoup.trace.explain`` — the same call the dashboard renders —
    so the terminal and the screen cannot describe the same payment differently. See
    docs/DECISIONS.md ADR-009.
    """
    trace = explain(event_from_entity(payment))
    colour = {"pass": GREEN, "refuse": GREEN, "defer": YELLOW, "info": ""}

    for index, step in enumerate(trace.steps, start=1):
        print()
        print(f"{BOLD}{index}. {step.title}{OFF}")
        show("verdict", step.verdict, colour.get(step.kind, ""))
        for label, value in step.fields:
            show(label, value)
        print(f"  {DIM}{step.detail}{OFF}")

    if trace.notes:
        print()
        print(f"{BOLD}What this payload could not tell us{OFF}")
        for note in trace.notes:
            print(f"  {DIM}- {note}{OFF}")


def main() -> int:
    print(f"{BOLD}Recoup - Razorpay sandbox demo{OFF}")
    print(f"{DIM}Test mode. No real money moves.{OFF}\n")

    try:
        client = RazorpayClient.from_env()
    except Exception as exc:  # noqa: BLE001 - surface any credential problem plainly
        print(f"Could not create a client: {exc}")
        return 1

    show("key", client.key_id[:14] + "...")

    seen_before = {p["id"] for p in client.list_payments(count=20)}

    order = client.create_order(
        amount_paise=AMOUNT_PAISE,
        receipt=f"recoup_{int(time.time())}",
        notes={"source": "recoup", "demo": "sandbox"},
    )
    show("order created", str(order["id"]), GREEN)

    CHECKOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKOUT_PATH.write_text(
        CHECKOUT_TEMPLATE.format(
            key_id=client.key_id,
            order_id=order["id"],
            amount=f"{AMOUNT_PAISE / 100:,.0f}",
        ),
        encoding="utf-8",
    )

    print()
    print(f"{BOLD}Opening checkout in your browser.{OFF}")
    print("  Make it fail on purpose, whichever method your account offers:")
    print(f"    1. {BOLD}Netbanking{OFF} - any bank, then click 'Failure' on the mock page")
    print(f"    2. {BOLD}Card{OFF} - any test card, then an OTP of fewer than 4 digits")
    print(f"    3. {BOLD}UPI{OFF} - enter failure@razorpay")
    print(f"  {DIM}{CHECKOUT_PATH.resolve()}{OFF}")
    webbrowser.open(CHECKOUT_PATH.resolve().as_uri())

    print()
    print(f"{DIM}Waiting for a failed payment", end="", flush=True)
    for _ in range(POLL_LIMIT):
        time.sleep(POLL_SECONDS)
        print(".", end="", flush=True)
        for payment in client.payments_for_order(str(order["id"])):
            if payment.get("status") == "failed" and payment["id"] not in seen_before:
                print(f"{OFF}\n")
                print(f"{GREEN}{BOLD}Failed payment received from Razorpay.{OFF}")
                describe_failure(payment)
                print()
                print(f"{DIM}That decline reason came from Razorpay's API, not a fixture.{OFF}")
                return 0

    print(f"{OFF}\n")
    print("No failed payment arrived. Re-run and use failure@razorpay at checkout.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

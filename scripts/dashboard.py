"""Run the demo dashboard and the webhook receiver on one port.

Run:  python scripts/dashboard.py           then open http://127.0.0.1:8000

The webhook endpoint is at ``/webhook/razorpay`` on the same server. Pointing Razorpay at
it needs a public tunnel; the dashboard's *Pull from Razorpay sandbox* button fetches the
same failed payments from the API instead, which is the path the demo uses because a
tunnel is one more thing that can fail on camera.

Binds to loopback by default. This process holds test-mode Razorpay credentials and an
unauthenticated control surface that can trigger outbound calls, so it has no business
listening on a public interface. ``--host`` will override that if you know why you want to.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from recoup.config import Settings

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
OFF = "\033[0m"

RESULTS = Path("data/results/eval.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install the api extra:")
        print('  .venv\\Scripts\\python.exe -m pip install -e ".[api]"')
        return 1

    from recoup.dashboard import create_dashboard

    settings = Settings.from_env()
    url = f"http://{args.host}:{args.port}"

    print(f"{BOLD}Recoup dashboard{OFF}")
    print(f"  {'dashboard':<22} {GREEN}{url}{OFF}")
    print(f"  {'webhook endpoint':<22} {url}/webhook/razorpay")

    if settings.razorpay_webhook_secret:
        print(f"  {'webhook signatures':<22} {GREEN}verified{OFF}")
    else:
        # Not a warning about tidiness. Without a secret the receiver refuses every
        # delivery, which is the correct behaviour and looks like a broken demo.
        print(f"  {'webhook signatures':<22} {YELLOW}NO SECRET SET - deliveries refused{OFF}")

    if settings.has_razorpay:
        print(f"  {'razorpay':<22} {GREEN}test keys loaded{OFF}")
    else:
        print(f"  {'razorpay':<22} {YELLOW}not configured - sandbox pull disabled{OFF}")

    if RESULTS.exists():
        print(f"  {'measured results':<22} {GREEN}{RESULTS}{OFF}")
    else:
        print(f"  {'measured results':<22} {YELLOW}missing - run scripts/run_eval.py{OFF}")

    print(f"\n{DIM}Ctrl-C to stop.{OFF}\n")

    if not args.no_open:
        webbrowser.open(url)

    uvicorn.run(create_dashboard(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

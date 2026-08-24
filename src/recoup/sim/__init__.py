"""Transaction simulator and recovery-episode mechanics.

See docs/DECISIONS.md ADR-002 for why evaluation happens in a simulator, and
src/recoup/sim/outcomes.py for why recovery is modelled by mechanism rather than
by fitted curve.
"""

from recoup.sim.entities import (
    Attempt,
    Bank,
    Contact,
    ContactChannel,
    Customer,
    FailedPayment,
    Instrument,
)
from recoup.sim.episode import (
    Action,
    ActionKind,
    EpisodeResult,
    EpisodeState,
    Policy,
    run_episode,
)
from recoup.sim.generator import Population, generate_population
from recoup.sim.outcomes import OutcomeModel, balance_fraction
from recoup.sim.params import SWEPT_PARAMETERS, SimParams, SweepRange
from recoup.sim.rails import Outage, RailHealth, generate_outages

__all__ = [
    "SWEPT_PARAMETERS",
    "Action",
    "ActionKind",
    "Attempt",
    "Bank",
    "Contact",
    "ContactChannel",
    "Customer",
    "EpisodeResult",
    "EpisodeState",
    "FailedPayment",
    "Instrument",
    "Outage",
    "OutcomeModel",
    "Policy",
    "Population",
    "RailHealth",
    "SimParams",
    "SweepRange",
    "balance_fraction",
    "generate_outages",
    "generate_population",
    "run_episode",
]

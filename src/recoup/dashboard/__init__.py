"""The demo surface: a read model over the engine, and nothing more.

See ``recoup.dashboard.app`` for why this holds no decision logic of its own.
"""

from recoup.dashboard.app import DashboardState, FeedEntry, create_dashboard

__all__ = ["DashboardState", "FeedEntry", "create_dashboard"]

"""Time-optimal kinodynamic agent.

Phase-space A* with gravity-aware ATSP pickup ordering. The planner is
decomposed into two levels:

1. Combinatorial: pick an order for the level's pickups that minimises total
   asymmetric travel time. Gravity makes the pair-cost matrix asymmetric (up
   is slower than down), so Held-Karp over the asymmetric TSP, not the
   symmetric detour-length TSP the Rust pathfinder solves.
2. Per-leg: weighted A* in phase space (position, velocity, rotation,
   pickup-bits) from the state at the end of the previous leg to collection
   of the next pickup. Momentum carries forward between legs naturally.

See ``solver.py`` for the implementation and ``agent.py`` for the ``BaseAgent``
wrapper consumed by ``run.py --agent kinodyn``.
"""

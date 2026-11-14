# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The network of flights of OpenFlights (openflights.org): its route database, 67,663 routes of airlines between
3,425 airports, as they were in June 2014. A route is one airline flying from one airport to another; a line of
routes.dat gives the airline, the code of the airport it leaves from and of the one it flies to, and the plane. The
database is under the Open Database License 1.0 (ODbL), which asks that a database made from it is shared under that
license too, so the course hosts none of it (course/route/bronnen.yaml, openflights-routes): everything_ai.hub
downloads routes.dat once, 2.4 MB, at a fixed commit of the repository of OpenFlights, and keeps it only when its
SHA-256 is the pinned one. `routes` reads it as CSV, so nothing in it is ever run. The numbers and pictures the
lessons make from it contain information from OpenFlights, made available under the ODbL."""
import csv
import functools

from everything_ai import hub

ROUTES = ("github.com/jpatokal/openflights", "routes.dat")

# The city of each airport a lesson names, by its code: routes.dat gives only the code.
CITIES = {
    "AMS": "Amsterdam", "FRA": "Frankfurt", "CDG": "Paris", "IST": "Istanbul", "ATL": "Atlanta", "PEK": "Beijing",
    "ORD": "Chicago", "MUC": "Munich", "DME": "Moscow", "DFW": "Dallas-Fort Worth", "DXB": "Dubai", "LHR": "London",
    "ABI": "Abilene",
}


def ready() -> bool:
    """Whether routes.dat is downloaded."""
    return hub.pinned(*ROUTES) is not None


@functools.cache
def routes() -> tuple[tuple[str, str], ...]:
    """Every route as (from, to), the codes of its two airports, in the order of the file. It raises
    FileNotFoundError when routes.dat is not downloaded yet: hub.download(*ROUTES) gets it."""
    path = hub.pinned(*ROUTES)
    if path is None:
        raise FileNotFoundError("routes.dat is not downloaded yet: everything_ai.hub.download(*everything_ai.flights.ROUTES)")
    with open(path, encoding="utf-8", newline="") as f:
        return tuple((row[2], row[4]) for row in csv.reader(f))


def graph() -> dict[str, set[str]]:
    """The network without directions, made anew at every call: every airport and the set of airports it has a route
    to or from. The one route from an airport to itself is left out."""
    out: dict[str, set[str]] = {}
    for a, b in routes():
        if a == b:
            continue
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out

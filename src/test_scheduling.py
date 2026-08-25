"""Tests for the alternative-time search used on scheduling conflicts.

`find_earlier_time` / `find_later_time` answer "when could this activity go
instead" when `find_conflicting_activity` rejects a slot. Both are pure
functions over an in-memory schedule, so they are the one part of `tools.py`
that can be tested without an LLM, a database or `data/therapy.json` — hence
this file, next to the scenario batch runner in `test.py` rather than in place
of it.

Run: `python test_scheduling.py` (also collectable by pytest).

The core of the suite is `_brute_force_earlier` / `_brute_force_later`: the
optimal placement is always flush against the start (resp. the end) of a
*blocking* activity, so scanning every minute of the day gives the exact answer
the anchor-based implementation must reproduce. That is what pins the bug both
functions had: they anchored on the wrong activity, which stays conflict-free —
and so looks correct — while sitting needlessly far from the requested time.
"""

import io
import json
import os
import random

import tools

DAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
DAY_MINUTES = 24 * 60


def act(name, time, duration, days=None, **extra):
    return {
        "activity_id": name.lower(),
        "name": name,
        "time": time,
        "duration_minutes": duration,
        "day_of_week": list(days) if days else list(DAYS),
        **extra,
    }


# A schedule shaped like the conversation that surfaced the bug: a midday meal
# and an evening one, hours apart.
MEALS = [act("Lunch", "12:30", 60), act("Dinner", "19:00", 60)]

# The real thing, copied from `scenarios/33.json` (day numbering is 1=Monday).
# `test_scenario_33_schedule_still_matches_the_file` fails if the scenario drifts.
SATURDAY = 6
SCENARIO_33_SCHEDULE = [
    act("Breakfast", "08:00", 20, days=range(1, 8)),
    act("Lunch", "12:30", 45, days=range(1, 8)),
    act("Dinner", "19:00", 45, days=range(1, 8)),
]


def _fits_in_day(activity):
    start = tools.hhmm_to_minutes(activity["time"])
    return start + activity["duration_minutes"] <= DAY_MINUTES


def _free_at(activity, schedule, start_minutes):
    probe = {**activity, "time": tools.minutes_to_hhmm(start_minutes)}
    return tools.find_conflicting_activity(probe, schedule) is None


def _brute_force_earlier(activity, schedule):
    """Latest start at or before the current one, by scanning every minute."""
    duration = activity["duration_minutes"]
    current_start = tools.hhmm_to_minutes(activity["time"])
    for start in range(current_start, -1, -1):
        if start + duration <= DAY_MINUTES and _free_at(activity, schedule, start):
            return tools.minutes_to_hhmm(start)
    return None


def _brute_force_later(activity, schedule):
    """Earliest start strictly after the current one, by scanning every minute."""
    duration = activity["duration_minutes"]
    current_start = tools.hhmm_to_minutes(activity["time"])
    for start in range(current_start + 1, DAY_MINUTES - duration + 1):
        if _free_at(activity, schedule, start):
            return tools.minutes_to_hhmm(start)
    return None


# --------------------------------------------------------------------------
# Regression cases: expected values written by hand, not by the implementation.
# (label, activity, schedule, expected earlier, expected later)
# --------------------------------------------------------------------------

REGRESSION_CASES = [
    (
        # The reported bug. Physio is longer than its overlap with Lunch, so
        # Lunch ends *before* Physio would: it used to be dropped from the
        # candidates and Dinner became the anchor -> "postpone to 20:00".
        "straddles lunch, longer than the overlap",
        act("Physio", "12:00", 120),
        MEALS,
        "10:30",
        "13:30",
    ),
    (
        # Same shape, short enough that Lunch survived the old filter: this one
        # already answered 13:30, which is why the bug read as intermittent.
        "ends inside lunch",
        act("Physio", "12:00", 60),
        MEALS,
        "11:30",
        "13:30",
    ),
    (
        # Starts inside Lunch. The old earlier-filter kept Lunch (it starts
        # before Physio), so this direction was already right.
        "starts inside lunch",
        act("Physio", "13:00", 60),
        MEALS,
        "11:30",
        "13:30",
    ),
    (
        # Only the tail overlaps. Anticipating must reach 12:00 (flush against
        # Lunch), not 11:30.
        "tail overlap only",
        act("Physio", "13:20", 30),
        MEALS,
        "12:00",
        "13:30",
    ),
    (
        # 13:30-13:45 is free but too short: postponing must skip to the end of
        # Rest, and must not offer 13:30.
        "first gap too short for the duration",
        act("Physio", "12:00", 120),
        [
            act("Lunch", "12:30", 60),
            act("Rest", "13:45", 30),
            act("Dinner", "19:00", 60),
        ],
        "10:30",
        "14:15",
    ),
    (
        # Nothing fits after Walk without crossing midnight: the historical
        # answer here was "24:00", a start time the tools' own validation
        # rejects, so `None` is the contract.
        "no room left before midnight",
        act("Walk", "22:30", 120),
        [act("Dinner", "22:00", 60)],
        "20:00",
        None,
    ),
    (
        # Scenario 33, the conversation that surfaced the bug: a Saturday lunch
        # asked for at 13:00 against Lunch 12:30-13:15 and Dinner 19:00-19:45.
        # Lunch ends before 14:00, so it used to be dropped and Dinner became
        # the anchor: the caregiver was offered "postpone to 19:45", hours away,
        # next to a correct "anticipate to 11:30". Days are integers here
        # because that is what the scenario files carry — both functions only
        # intersect the sets, so either representation must work.
        "scenario 33: saturday lunch against Lunch and Dinner",
        act("Low-sodium salad lunch", "13:00", 60, days=[SATURDAY]),
        SCENARIO_33_SCHEDULE,
        "11:30",
        "13:15",
    ),
    (
        # Nap blocks; Lunch is on another day. Lunch sits exactly where it would
        # change the answer if the day filter were dropped (9:30 instead of
        # 11:00), so this case fails loudly if it ever is.
        "an activity on other days must not narrow the window",
        act("Physio", "12:00", 120, days=["tuesday"]),
        [
            act("Lunch", "11:30", 60, days=["monday"]),
            act("Nap", "13:00", 60, days=["tuesday"]),
        ],
        "11:00",
        "14:00",
    ),
    (
        # Same idea for validity periods: Nap blocks, Lunch shares the days but
        # not the months. Honouring its window would answer 10:15.
        "an activity outside the validity period must not narrow the window",
        act("Physio", "12:00", 60, valid_from="2026-01-01", valid_until="2026-01-31"),
        [
            act("Lunch", "11:00", 60, valid_from="2026-03-01", valid_until="2026-03-31"),
            act("Nap", "12:15", 60, valid_from="2026-01-01", valid_until="2026-01-31"),
        ],
        "11:15",
        "13:15",
    ),
]


def test_regression_expected_times():
    for label, activity, schedule, want_earlier, want_later in REGRESSION_CASES:
        earlier = tools.find_earlier_time(dict(activity), schedule)
        later = tools.find_later_time(dict(activity), schedule)
        assert earlier == want_earlier, f"{label}: earlier {earlier} != {want_earlier}"
        assert later == want_later, f"{label}: later {later} != {want_later}"


def test_regression_matches_brute_force():
    for label, activity, schedule, _, _ in REGRESSION_CASES:
        earlier = tools.find_earlier_time(dict(activity), schedule)
        later = tools.find_later_time(dict(activity), schedule)
        assert earlier == _brute_force_earlier(activity, schedule), (
            f"{label}: earlier is not the latest free slot"
        )
        assert later == _brute_force_later(activity, schedule), (
            f"{label}: later is not the earliest free slot"
        )


def test_suggestions_are_applicable():
    """A suggested time must be free, inside the day, and on the right side."""
    for label, activity, schedule, _, _ in REGRESSION_CASES:
        current_start = tools.hhmm_to_minutes(activity["time"])
        earlier = tools.find_earlier_time(dict(activity), schedule)
        later = tools.find_later_time(dict(activity), schedule)
        for direction, suggested in (("earlier", earlier), ("later", later)):
            if suggested is None:
                continue
            start = tools.hhmm_to_minutes(suggested)
            assert 0 <= start < DAY_MINUTES, f"{label}: {direction} {suggested} invalid"
            assert start + activity["duration_minutes"] <= DAY_MINUTES, (
                f"{label}: {direction} {suggested} runs past midnight"
            )
            assert _free_at(activity, schedule, start), (
                f"{label}: {direction} {suggested} conflicts"
            )
        if earlier is not None:
            assert tools.hhmm_to_minutes(earlier) <= current_start, (
                f"{label}: earlier moved forward"
            )
        if later is not None:
            assert tools.hhmm_to_minutes(later) > current_start, f"{label}: later did not move"


def test_degenerate_durations_return_the_current_time():
    for duration in (0, -30):
        activity = act("Physio", "12:00", duration)
        assert tools.find_earlier_time(dict(activity), MEALS) == "12:00"
        assert tools.find_later_time(dict(activity), MEALS) == "12:00"


def test_fully_booked_day_has_no_alternative():
    schedule = [act("Busy", "00:00", DAY_MINUTES)]
    activity = act("Physio", "12:00", 60)
    assert tools.find_earlier_time(dict(activity), schedule) is None
    assert tools.find_later_time(dict(activity), schedule) is None


def _random_schedule(rng, size):
    schedule = []
    for i in range(size):
        start = rng.randrange(0, DAY_MINUTES - 15, 15)
        duration = min(rng.choice([15, 30, 45, 60, 90, 120]), DAY_MINUTES - start)
        days = rng.sample(DAYS, rng.randint(1, len(DAYS)))
        schedule.append(act(f"A{i}", tools.minutes_to_hhmm(start), duration, days=days))
    return schedule


def test_random_schedules_match_brute_force():
    """The property that pins both bugs: same answer as a minute-by-minute scan.

    A wrong anchor still yields a conflict-free time, so only optimality
    separates the fixed implementation from the broken one.
    """
    rng = random.Random(20260825)
    checked = 0
    conflicting = 0
    for _ in range(600):
        schedule = _random_schedule(rng, rng.randint(1, 6))
        start = rng.randrange(0, DAY_MINUTES - 15, 15)
        duration = rng.choice([15, 30, 45, 60, 90, 120, 180])
        activity = act(
            "New",
            tools.minutes_to_hhmm(start),
            duration,
            days=rng.sample(DAYS, rng.randint(1, len(DAYS))),
        )
        if not _fits_in_day(activity):
            continue
        checked += 1
        # Only the conflict path is reachable in production: both functions are
        # called from `find_scheduling_conflicts` after a clash was found. On a
        # free slot the minute-scan answer for `later` is current + 1 minute,
        # which the anchor design deliberately never proposes — see
        # `test_free_slot_needs_no_alternative` for the contract there.
        if tools.find_conflicting_activity(activity, schedule) is None:
            continue
        conflicting += 1
        context = (
            f"activity={activity['time']}/{duration}min "
            f"days={len(activity['day_of_week'])} "
            f"schedule={[(a['time'], a['duration_minutes']) for a in schedule]}"
        )
        assert tools.find_earlier_time(dict(activity), schedule) == _brute_force_earlier(
            activity, schedule
        ), f"earlier disagrees: {context}"
        assert tools.find_later_time(dict(activity), schedule) == _brute_force_later(
            activity, schedule
        ), f"later disagrees: {context}"
    assert checked > 400, f"only {checked} random cases were usable"
    assert conflicting > 100, f"only {conflicting} random cases actually conflicted"


def test_free_slot_needs_no_alternative():
    """The unreachable branch, pinned so the anchor set stays honest.

    `find_earlier_time` used to answer `current_start - duration` here: its
    "stay put" anchor was the start of the requested slot instead of its end,
    so it shifted the activity a whole duration back with nothing in the way.
    `find_later_time` has no such anchor by design — postponing means moving —
    so it either finds a genuine later anchor or nothing.
    """
    for label, schedule in (
        ("empty schedule", []),
        ("no overlap", [act("Lunch", "12:30", 60)]),
        ("other days only", [act("Lunch", "12:00", 60, days=["monday"])]),
    ):
        activity = act("Physio", "10:00", 60, days=["tuesday"])
        assert tools.find_conflicting_activity(activity, schedule) is None, label
        assert tools.find_earlier_time(dict(activity), schedule) == "10:00", label
        later = tools.find_later_time(dict(activity), schedule)
        if later is not None:
            start = tools.hhmm_to_minutes(later)
            assert start > tools.hhmm_to_minutes(activity["time"]), label
            assert _free_at(activity, schedule, start), label


def test_conflict_report_offers_the_nearest_alternatives():
    """`find_scheduling_conflicts` is what the caregiver actually reads."""
    result = tools.find_scheduling_conflicts(act("Physio", "12:00", 120), MEALS)
    assert result is not None
    assert result["issue"] == tools.ISSUE_SCHEDULE_CONFLICT
    message = result["message"]
    assert "Lunch" in message, "the report must name the activity it clashed with"
    assert "Anticipate the activity at 10:30" in message
    assert "Postpone the activity at 13:30" in message
    assert "20:00" not in message, "must not anchor on the unrelated Dinner"
    assert "24:00" not in message


def test_conflict_report_absent_when_the_slot_is_free():
    assert tools.find_scheduling_conflicts(act("Physio", "10:00", 60), MEALS) is None


def test_conflict_report_when_nothing_fits():
    schedule = [act("Busy", "00:00", DAY_MINUTES)]
    result = tools.find_scheduling_conflicts(act("Physio", "12:00", 60), schedule)
    assert result is not None
    assert "not possible alternative time" in result["message"]


def test_scenario_33_schedule_still_matches_the_file():
    """Keep the hardcoded copy honest, without depending on the file existing."""
    path = os.path.join(os.path.dirname(__file__), "..", "scenarios", "33.json")
    if not os.path.exists(path):
        return
    with io.open(path, encoding="utf-8") as handle:
        activities = json.load(handle)["activities"]
    keys = ("name", "time", "duration_minutes", "day_of_week")
    actual = sorted(tuple(str(a[k]) for k in keys) for a in activities)
    expected = sorted(tuple(str(a[k]) for k in keys) for a in SCENARIO_33_SCHEDULE)
    assert actual == expected, f"scenarios/33.json changed: {actual} != {expected}"


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL {name}\n     {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Cascade Intelligence System for Bengaluru Traffic Operations
**ASTRAM Platform — Pre-emptive Deployment Engine**

---

## THE INCIDENT

On December 16, 2023, a single pothole was reported on Bannerghatta Road at the start of the evening commute. Within two hours, 30 vehicles broke down at that location — each one blocking a lane, each one requiring a separate dispatch call, each one making the corridor worse for every vehicle behind it.

## THE PATTERN

When a pothole or waterlogging event appears on a high-traffic corridor, it does not stay a single incident — it triggers a chain of breakdowns, road closures, and diversions that overwhelm the corridor within minutes. We analysed 8,173 real ASTRAM events from Bengaluru (November 2023 – April 2024) and found 176 such chain-trigger events across 14 named corridors, each one producing an average of 8.5 follow-on incidents before the first responder arrived.

## HOW THE SYSTEM WORKS

The system reads every incoming event from the ASTRAM platform — the corridor name, the incident type, the time of day, and whether it falls on a peak-hour route. It then checks that event against the historical pattern of every past cascade on that corridor: does this type of incident, at this hour, on this road, match the signature of a chain reaction? If it does, the system fires a deployment alert within 30 seconds — not a chart, not a forecast, but a specific instruction: how many officers, what equipment, which corridor, and an estimated time to clear.

## WHAT AN OFFICER SEES

When a high-risk event is detected, the duty officer's screen displays a single alert card with a red cascade risk indicator and the corridor name in bold. Below it, a plain-English deployment instruction reads, for example: *"Deploy 8 traffic police, 2 breakdown recovery units, and 4 barricades to Hosur Road — estimated clearance: 45 minutes."* The card also shows the primary reason for the alert in language an officer can act on — such as *"Waterlogging on this corridor during evening peak has triggered lane closures 6 out of 8 times in the past 4 months."* There is no probability score, no graph to interpret, and no second screen to navigate. The officer reads the card, confirms the dispatch, and the units move.

## RESULTS

| Finding | Value |
|---|---|
| Cascade events identified | 176 seeds → 1,499 secondary events across 14 corridors |
| Highest-risk corridor | Hosur Road — 52 cascade triggers, 558 secondary events in 4 months |
| Response improvement | Reactive 30+ min dispatch → Pre-emptive alert fires within seconds of event registration |

## WHAT MAKES THIS DIFFERENT

Today, every incident on a corridor is treated as a standalone event and dispatched after it is reported — this system treats the *first* incident as an early warning and deploys resources *before* the corridor collapses.

## NEXT 90 DAYS

With a live connection to the ASTRAM event feed, the system moves from historical analysis to real-time corridor monitoring — every new event is scored the moment it is logged, and deployment alerts are pushed directly to officers via WhatsApp with a one-tap confirmation. A feedback loop from field teams — actual clearance times, resources used, incidents that did not cascade — tightens the system's accuracy with every shift.

---

*Built on 8,173 verified ASTRAM platform events. Demo corridor: Mysore Road, March 7, 2024 — a waterlogging event at 10:51 IST triggered 8 incidents in 76 minutes, 6 requiring road closure. The system would have fired a pre-deployment alert at 10:51.*

"""The eval suite: seeded floods crossed with situations.

dev is for prompt work; test is held out and only run to report. Seeds are disjoint from the
benchmark (100s, 200s), the GP calibration (3000s) and the demo.
"""

from dataclasses import dataclass

from kipimo import config

SITUATIONS = {
    "calm": {"profile": "calm", "style": "template", "faults": ()},
    "messy": {"profile": "calm", "style": "freeform", "faults": ()},
    "hostile": {"profile": "adversarial", "style": "freeform", "faults": ()},
    "storm": {"profile": "calm", "style": "freeform", "faults": (), "weather": True},
    "faults": {
        "profile": "calm",
        "style": "freeform",
        "faults": (("degrade", 500.0), ("offline", 700.0), ("online", 1600.0)),
    },
}
SPLITS = {  # broad flood seeds, river corridor seeds
    "dev": (range(300, 303), range(310, 313)),
    "test": (range(400, 406), range(410, 416)),
}


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    seed: int
    situation: str


def suite(split: str) -> list[Case]:
    broad, corridor = SPLITS[split]
    return [
        Case(f"{fam}-{seed}-{sit}", fam, seed, sit)
        for fam, seeds in (("broad", broad), ("corridor", corridor))
        for seed in seeds
        for sit in SITUATIONS
    ]


MAST = {"x": 512.0, "y": 512.0, "r": 40.0, "name": "mast"}  # mid farm, so plans must skirt it


def config_for(case: Case, model: str | None = None) -> config.Config:
    s = SITUATIONS[case.situation]
    world = {"flood_stage": [1.0, 1.7]} if case.family == "corridor" else {}
    schedule = [{"t": t, "state": k} for k, t in s["faults"] if k in ("offline", "online")]
    agent = {"sms_profile": s["profile"], "sms_style": s["style"]} | (
        {"model": model} if model else {}
    )
    return config.load(
        None,
        seed=case.seed,
        world=world,
        belief={"model": "gp"},
        agent=agent,
        uplink={"schedule": schedule},
        weather={"enabled": bool(s.get("weather"))},
        envelope={"no_fly": [MAST]},
    )


def injector(case: Case):
    """Setup hook that degrades the camera mid mission, as the dashboard button does."""
    at = [t for k, t in SITUATIONS[case.situation]["faults"] if k == "degrade"]

    def setup(st) -> None:
        def tick() -> None:
            if at and st.ep.bus.t >= at[0]:
                at.pop()
                st.ep.sensor.degraded = True
                st.ep.bus.emit("SENSOR_STATE", degraded=True)

        st.ep.tickers.append(tick)

    return setup

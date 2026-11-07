from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


class WorldCfg(BaseModel):
    size: int = 256
    cell_m: float = 4.0
    n_parcels: tuple[int, int] = (12, 20)
    flood_stage: tuple[float, float] = (1.6, 2.6)  # river stage in metres; higher floods more
    terrain: str = "synthetic"  # synthetic | tana, real Copernicus relief on the Tana floodplain


class RobotCfg(BaseModel):
    battery_wh: float = 200.0
    hover_w: float = 330.0  # rotor power in still air; 8 m/s cruise draws 253 W
    max_airspeed: float = 15.0
    tilt_deg: float = 25.0  # thrust tilt limit, caps acceleration at g tan(tilt)
    speed_mps: float = 8.0
    altitude_m: float = 40.0
    fov_deg: float = 80.0
    reserve_wh: float = 12.0


class GPCfg(BaseModel):
    """Prior sds and residual blocks of the GP belief, fitted by kipimo.belief.fit."""

    s_bias: float = 0.03
    s_prior: float = 0.1
    s_resp: float = 0.06
    s_parcel: float = 0.03
    blocks: list[tuple[float, float]] = [(0.0035, 14.0), (0.002, 230.0), (0.001, 2000.0)]
    sigma_sys: float = 0.02  # measurement error that repeat looks do not average away
    bias_frac: float = 0.2  # parcel shared measurement bias, as a fraction of its noise sd


class BeliefCfg(BaseModel):
    model: str = "grid"  # grid | gp
    gp: GPCfg = GPCfg()
    k: float = 12.0  # pseudo counts per observation at quality 1
    k0: float = 2.0  # satellite prior pseudo counts
    h: float = 30.0  # evidence at which a cell is half trusted over the prior
    rho: float = 0.3  # share of per cell measurement noise that is systematic
    sigma_extrap: float = 0.07  # parcel level sd of the radar prior, measured 0.05 plus margin
    ref_width: float = 0.6  # CI width that counts as zero confidence
    corr_cells: float = 160.0  # cells per spatially independent patch


class Zone(BaseModel):
    x: float
    y: float
    r: float
    name: str = "no_fly"


class EnvelopeCfg(BaseModel):
    geofence_margin_m: float = 40.0
    zone_margin_m: float = 6.0
    no_fly: list[Zone] = []


class PlannerCfg(BaseModel):
    fixed_cost_m: float = 60.0  # per lane overhead for turns and settling
    min_gain_per_m: float = 0.2  # stop when a metre of flight buys less risk reduction than this


class AgentCfg(BaseModel):
    model: str = "gemini-4-argon"  # the Adjuster; reads GEMINI_API_KEY
    roles: dict[str, str] = {  # per-role model, else `model`
        "field": "gemini-robotics-er-2-preview",
        "farmer": "gemini-3.8-flash",
    }
    thinking: dict[str, str] = {"field": "medium", "adjuster": "medium", "farmer": "low"}
    max_tool_calls: int = 45  # enough to survey, claim every field and answer new events
    max_seconds: float = 300.0  # per role; Gemini 4 Argon thinks 4 to 11 s a turn
    mission_seconds: float = 1200.0  # wall clock for every role together; then rules finish
    max_sortie_m: float = 3000.0  # one battery pack per objective
    sms_profile: str = "calm"  # calm | adversarial
    sms_style: str = "template"  # template | freeform
    field_wakes: int = 4  # times the field agent is woken by new events
    role_calls: int = 4  # tool calls per farmer message or claim review
    memory: str = ""  # JSON file that outlives missions; empty means none


class UplinkCfg(BaseModel):
    state: str = "online"
    loss_pct: float = 30.0
    latency_ms: float = 800.0
    db: str = ":memory:"
    schedule: list[dict] = []  # [{t: seconds, state: online|degraded|offline}]


class SimCfg(BaseModel):
    dt: float = 1.0
    sensor: str = "sim"  # sim | classical | onnx


class WeatherCfg(BaseModel):
    enabled: bool = False
    limit: float = 10.0  # wind the drone may fly in, m/s
    hours: int = 24
    members: int = 32
    calm: float = 4.0
    storm_peak: float = 12.0  # extra wind at the storm's peak
    storm_hour: float = 1.0
    storm_width: float = 1.2
    spread: float = 0.8
    breeze: float = 3.0  # mean wind when no forecast is loaded, m/s
    gust: float = 0.2  # gust sd as a share of the mean wind
    gust_s: float = 4.0  # gust correlation time


class FloodCfg(BaseModel):
    enabled: bool = True
    gauge: str = "Tana at Garsen"
    hours: int = 168  # forecast horizon, as Flood Hub's seven days
    issued_hour: int = 6  # local clock
    lead_days: tuple[int, int] = (1, 5)
    base_m: float = 2.0
    warning: float = 3.2
    danger: float = 4.0
    peak_m: float = 4.8
    rise_m_per_h: float = 0.08


class ApprovalCfg(BaseModel):
    mode: str = "human"  # human: the actuary signs off every claim | auto: rules pass clean ones
    actuary: str = "duty actuary"  # who signs; a deployment takes this from the login


class Config(BaseModel):
    seed: int = 7
    world: WorldCfg = WorldCfg()
    robot: RobotCfg = RobotCfg()
    belief: BeliefCfg = BeliefCfg()
    envelope: EnvelopeCfg = EnvelopeCfg()
    planner: PlannerCfg = PlannerCfg()
    agent: AgentCfg = AgentCfg()
    uplink: UplinkCfg = UplinkCfg()
    sim: SimCfg = SimCfg()
    weather: WeatherCfg = WeatherCfg()
    flood: FloodCfg = FloodCfg()
    approval: ApprovalCfg = ApprovalCfg()


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(scenario: str | None = None, **overrides) -> Config:
    raw = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    if scenario:
        raw = _merge(
            raw, yaml.safe_load((CONFIG_DIR / "scenarios" / f"{scenario}.yaml").read_text()) or {}
        )
    raw = _merge(raw, {k: v for k, v in overrides.items() if v is not None})
    return Config.model_validate(raw)

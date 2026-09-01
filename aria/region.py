from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

STATE_NAMES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
}
STATE_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20",
    "KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28",
    "MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
    "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45",
    "SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
    "WI":"55","WY":"56","DC":"11",
}

CONTIGUOUS_STATES = tuple(
    s for s in STATE_NAMES if s not in {"AK", "HI", "DC"}
)

PRIMARY_STATES = ("MN","WI","IL","IA","ND","SD","NE")
RECTANGLE_STATES = (
    "MN","WI","IL","IA","ND","SD","NE","KS","MO","MI","IN","CO","WY","OK"
)

@lru_cache(maxsize=64)
def _infer_states(west: float, south: float, east: float, north: float):
    """Infer US states intersecting a bbox using Cartopy/Natural Earth.

    Falls back to all contiguous states if geographic features cannot be read.
    The final observations are still clipped to the requested region, so the
    fallback favors completeness over query efficiency.
    """
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
        shp = shpreader.natural_earth(
            resolution="50m", category="cultural",
            name="admin_1_states_provinces_lakes"
        )
        bbox = box(west, south, east, north)
        hits = []
        name_to_code = {v.lower(): k for k, v in STATE_NAMES.items()}
        for rec in shpreader.Reader(shp).records():
            a = rec.attributes
            if a.get("adm0_a3") != "USA" and a.get("admin") != "United States of America":
                continue
            if not rec.geometry.intersects(bbox):
                continue
            name = str(a.get("name") or a.get("name_en") or "").lower()
            code = a.get("postal") or a.get("iso_3166_2")
            if isinstance(code, str) and "-" in code:
                code = code.split("-")[-1]
            if code not in STATE_NAMES:
                code = name_to_code.get(name)
            if code in STATE_NAMES:
                hits.append(code)
        return tuple(sorted(set(hits))) or CONTIGUOUS_STATES
    except Exception:
        return CONTIGUOUS_STATES

@dataclass(frozen=True)
class Region:
    name: str = "GPGL"
    west: float = -104.1
    south: float = 36.9
    east: float = -86.7
    north: float = 49.1
    states: tuple[str, ...] = field(default_factory=tuple)
    query_states_override: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.west >= self.east:
            raise ValueError("Region west must be less than east.")
        if self.south >= self.north:
            raise ValueError("Region south must be less than north.")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("Longitude bounds must be between -180 and 180.")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("Latitude bounds must be between -90 and 90.")

    @property
    def lat_range(self):
        return (self.south, self.north)

    @property
    def lon_range(self):
        return (self.west, self.east)

    @property
    def query_states(self):
        if self.query_states_override:
            return self.query_states_override
        return _infer_states(self.west, self.south, self.east, self.north)

    @property
    def state_names(self):
        values = self.states or self.query_states
        return tuple(STATE_NAMES[s] for s in values if s in STATE_NAMES)

    @property
    def query_state_names(self):
        return tuple(STATE_NAMES[s] for s in self.query_states if s in STATE_NAMES)

    @property
    def state_fips(self):
        return tuple(STATE_FIPS[s] for s in self.query_states if s in STATE_FIPS)

    @property
    def asos_networks(self):
        return tuple(f"{s}_ASOS" for s in self.query_states)

    @property
    def airnow_bbox(self):
        return f"{self.west},{self.south},{self.east},{self.north}"

    @property
    def cache_key(self):
        safe = self.name.lower().replace(" ", "_").replace("/", "_")
        return f"{safe}-{self.west:.3f}-{self.south:.3f}-{self.east:.3f}-{self.north:.3f}"

    def contains(self, latitude, longitude):
        return (
            (latitude >= self.south) & (latitude <= self.north)
            & (longitude >= self.west) & (longitude <= self.east)
        )

    @classmethod
    def from_center_radius(
        cls,
        name,
        center_lat,
        center_lon,
        radius_km,
        **kwargs,
    ):
        """Build a bounding-box Region around a center point and radius."""
        center_lat = float(center_lat)
        center_lon = float(center_lon)
        radius_km = float(radius_km)

        if radius_km <= 0:
            raise ValueError("radius_km must be positive.")

        lat_delta = radius_km / 111.0
        coslat = max(math.cos(math.radians(center_lat)), 0.15)
        lon_delta = radius_km / (111.0 * coslat)

        return cls(
            name=name,
            west=center_lon - lon_delta,
            east=center_lon + lon_delta,
            south=center_lat - lat_delta,
            north=center_lat + lat_delta,
            **kwargs,
        )

    def as_dict(self):
        return {
            "name": self.name,
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "query_states": list(self.query_states),
        }


GPGL_REGION = Region(
    name="GPGL", west=-104.1, south=36.9, east=-86.7, north=49.1,
    states=PRIMARY_STATES, query_states_override=RECTANGLE_STATES,
)
SGP_REGION = Region.from_center_radius(
    name="ARM SGP", center_lat=36.605, center_lon=-97.485, radius_km=400.0
)
UPPER_MIDWEST_REGION = Region(name="Upper Midwest", west=-100.5, south=40.0, east=-84.0, north=50.0)
CONUS_REGION = Region(name="CONUS", west=-125.0, south=24.0, east=-66.5, north=50.0)

REGION_PRESETS = {
    "GPGL": GPGL_REGION,
    "ARM SGP": SGP_REGION,
    "Upper Midwest": UPPER_MIDWEST_REGION,
    "CONUS": CONUS_REGION,
}

def region_from_dict(payload):
    return Region(
        name=payload.get("name", "Custom"),
        west=float(payload["west"]), south=float(payload["south"]),
        east=float(payload["east"]), north=float(payload["north"]),
        query_states_override=tuple(payload.get("query_states", ())),
    )

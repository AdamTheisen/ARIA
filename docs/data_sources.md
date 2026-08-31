# ARIA Data Sources

Current ARIA workflows include:

- ASOS/AWOS/METAR surface observations via Iowa Environmental Mesonet
- AirNow PM2.5 and ozone
- NWS/IEM radiosondes
- SondeHub trajectories
- NEXRAD Level-II radar
- NOAA MRMS quality-controlled reflectivity
- NOAA HRRR through Herbie
- local ERF and E3SM scaffolds
- local authorized aircraft observations
- ARM/ACT-oriented integration points

All region-aware sources should consume the same `Region` object and clip data
to its bounding box. Regional NEXRAD discovery uses Py-ART's national site
metadata when available, replacing the old GPGL-only site assumption.

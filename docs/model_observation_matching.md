# Model / Observation Time Matching

For the default radiosonde-versus-HRRR workflow, ARIA:

1. Finds the newest available radiosonde in the active region.
2. Selects an HRRR initialization strictly earlier than the sonde launch.
3. Selects the forecast hour whose valid time is nearest the launch.
4. Reports sonde time, HRRR initialization, forecast hour, valid time, and time offset.

For a standard 12:00 UTC radiosonde, the expected default is typically the
11:00 UTC HRRR initialization at F01.

Manual model initialization and forecast-hour selection remains available for
forecast-skill experiments.

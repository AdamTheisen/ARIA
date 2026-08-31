from aria import GPGL_REGION
from aria.models import HRRRAdapter
from aria.plotting import plot_model_field

adapter = HRRRAdapter(GPGL_REGION)

# Latest conservatively available HRRR analysis.
ds, run = adapter.open_run(
    forecast_hour=0,
    product="sfc",
    variables=["air_temperature_2m"],
)

print(run.label)
print(ds)

fig = plot_model_field(
    ds,
    "air_temperature_2m",
    GPGL_REGION,
)
fig.savefig("output/hrrr_temperature.png", dpi=160, bbox_inches="tight")

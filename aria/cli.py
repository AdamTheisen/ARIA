from __future__ import annotations
import argparse
import json

from .cubes import CubeManager
from .recipe import ARIARecipe
from .history import historical_plan, backfill_storage, HISTORICAL_SOURCE_CAPABILITIES
from .region import REGION_PRESETS, region_from_dict
from .storage import (
    load_config, save_config, load_state, update_storage, status_rows,
    storage_size_bytes, region_store_root, migrate_legacy_snapshots,
    list_storage_profiles, create_storage_profile, set_active_profile,
    all_status_rows, total_storage_size_bytes,
)

def main(argv=None):
    p=argparse.ArgumentParser(
        prog="aria",
        description="ARIA — Atmospheric Regional Integration and Analysis",
    )
    s=p.add_subparsers(dest="cmd",required=True)

    c=s.add_parser("cube",help="Manage ARIA Icechunk/Zarr cubes")
    cs=c.add_subparsers(dest="action",required=True)
    for action in ("init","info","validate"):
        x=cs.add_parser(action)
        x.add_argument("name")

    r=s.add_parser("region",help="Inspect built-in ARIA regions")
    rs=r.add_subparsers(dest="action",required=True)
    rs.add_parser("list")
    show=rs.add_parser("show")
    show.add_argument("name",choices=list(REGION_PRESETS))

    recipe=s.add_parser("recipe",help="Validate an ARIA YAML cube recipe")
    recipe.add_argument("path")

    storage=s.add_parser("storage",help="Configure and run persistent ARIA regional storage")
    ss=storage.add_subparsers(dest="action",required=True)
    status=ss.add_parser("status",help="Show collector/storage status for all profiles")
    status.add_argument("--profile")
    update=ss.add_parser("update",help="Collect all enabled sources that are due across storage profiles")
    update.add_argument("--source",choices=["surface","mrms","hrrr","air_quality","radiosonde","sst","marine"])
    update.add_argument("--profile",help="Limit collection to one named storage profile")
    update.add_argument("--force",action="store_true",help="Ignore source cadence")
    enable=ss.add_parser("enable",help="Enable background collection for the active or named profile")
    enable.add_argument("--profile")
    disable=ss.add_parser("disable",help="Pause background collection for the active or named profile")
    disable.add_argument("--profile")
    migrate=ss.add_parser("migrate",help="Convert legacy NetCDF and snapshot Parquet into cloud-optimized ARIA storage")
    migrate.add_argument("--source",choices=["surface","mrms","hrrr","air_quality","radiosonde","sst","marine"])
    profiles=ss.add_parser("profiles",help="List named storage profiles")
    create=ss.add_parser("create-profile",help="Create a storage profile")
    create.add_argument("name")
    create.add_argument("--region",choices=list(REGION_PRESETS),default="GPGL")
    create.add_argument("--enable",action="store_true")
    select=ss.add_parser("select-profile",help="Choose the profile edited by default")
    select.add_argument("name")

    history=s.add_parser("history",help="Plan or run historical backfill into named ARIA stores")
    hs=history.add_subparsers(dest="action",required=True)
    plan=hs.add_parser("plan",help="Estimate historical retrieval steps")
    plan.add_argument("--profile",required=True)
    plan.add_argument("--start",required=True)
    plan.add_argument("--end",required=True)
    plan.add_argument("--sources",default="mrms,hrrr")
    plan.add_argument("--step-minutes",type=int)
    build=hs.add_parser("build",help="Backfill supported historical sources")
    build.add_argument("--profile",required=True)
    build.add_argument("--start",required=True)
    build.add_argument("--end",required=True)
    build.add_argument("--sources",default="mrms,hrrr")
    build.add_argument("--step-minutes",type=int)
    build.add_argument("--max-steps",type=int)

    a=p.parse_args(argv)

    if a.cmd=="region":
        if a.action=="list":
            for name,region in REGION_PRESETS.items():
                print(f"{name}: {region.west},{region.south},{region.east},{region.north}")
        else:
            print(json.dumps(REGION_PRESETS[a.name].as_dict(),indent=2))
        return

    if a.cmd=="recipe":
        rec=ARIARecipe.load(a.path)
        rec.validate()
        print(json.dumps({
            "name":rec.name,
            "region":rec.region.as_dict(),
            "sources":list(rec.sources),
            "storage":rec.storage,
            "compute":rec.compute,
        },indent=2,default=str))
        return

    if a.cmd=="history":
        sources=tuple(x.strip() for x in a.sources.split(",") if x.strip())
        if a.action=="plan":
            print(json.dumps(historical_plan(a.profile,a.start,a.end,sources,step_minutes=a.step_minutes),indent=2,default=str))
        else:
            print(json.dumps(backfill_storage(
                a.profile,a.start,a.end,sources=sources,
                step_minutes=a.step_minutes,max_steps=a.max_steps,
            ),indent=2,default=str))
        return

    if a.cmd=="storage":
        if a.action=="profiles":
            cfg=load_config()
            print(json.dumps({"active_profile":cfg.get("active_profile"),"profiles":list_storage_profiles(cfg)},indent=2,default=str))
            return
        if a.action=="create-profile":
            cfg=create_storage_profile(a.name,REGION_PRESETS[a.region],enabled=a.enable)
            print(json.dumps({"created":cfg.get("profile_name"),"region":cfg.get("region",{}).get("name"),"enabled":cfg.get("enabled")},indent=2))
            return
        if a.action=="select-profile":
            cfg=set_active_profile(a.name)
            print(f"Active storage profile: {cfg.get('profile_name')}")
            return
        if a.action in ("enable","disable"):
            cfg=load_config(a.profile) if a.profile else load_config()
            cfg["enabled"]=(a.action=="enable"); save_config(cfg)
            print(f"ARIA storage profile {cfg.get('profile_name')} " + ("enabled" if cfg["enabled"] else "paused"))
            return
        if a.action=="update":
            state=update_storage(source=a.source,force=a.force,profile=a.profile)
            print(json.dumps({"result":state.get("last_collector_result"),"finished":state.get("last_collector_finish_utc"),"sources":state.get("sources",{})},indent=2,default=str))
            return
        if a.action=="migrate":
            print(json.dumps(migrate_legacy_snapshots(source=a.source),indent=2,default=str))
            return
        cfg=load_config(a.profile) if getattr(a,"profile",None) else load_config()
        state=load_state(cfg.get("profile_name"))
        print(json.dumps({
            "active_profile":cfg.get("active_profile"),
            "profile":cfg.get("profile_name"),
            "enabled":cfg.get("enabled",False),
            "region":cfg.get("region",{}).get("name"),
            "root":str(region_store_root(cfg)),
            "size_bytes":storage_size_bytes(cfg),
            "total_size_bytes":total_storage_size_bytes(cfg),
            "collector_last_finish":state.get("last_collector_finish_utc"),
            "sources":status_rows(cfg,state),
        },indent=2,default=str))
        return

    m=CubeManager()
    if a.action=="init":
        m.init(a.name); print(f"Initialized {a.name}")
    elif a.action=="info":
        print(json.dumps(m.info(a.name),indent=2,default=str))
    else:
        i=m.info(a.name)
        assert i["variables"],"cube has no variables"
        print(f"{a.name}: OK")

if __name__=="__main__":
    main()

from __future__ import annotations
import argparse
import json

from .cubes import CubeManager
from .recipe import ARIARecipe
from .region import REGION_PRESETS

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

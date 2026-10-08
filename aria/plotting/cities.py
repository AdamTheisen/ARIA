from __future__ import annotations
US_CITIES=[
("Minneapolis",44.9778,-93.2650),("St. Paul",44.9537,-93.0900),("Duluth",46.7867,-92.1005),("Fargo",46.8772,-96.7898),("Grand Forks",47.9253,-97.0329),("Bismarck",46.8083,-100.7837),("Sioux Falls",43.5446,-96.7311),("Omaha",41.2565,-95.9345),("Lincoln",40.8136,-96.7026),("Des Moines",41.5868,-93.6250),("Madison",43.0731,-89.4012),("Milwaukee",43.0389,-87.9065),("Green Bay",44.5133,-88.0133),("Chicago",41.8781,-87.6298),("Kansas City",39.0997,-94.5786),("Wichita",37.6872,-97.3301),("Oklahoma City",35.4676,-97.5164),("Tulsa",36.1540,-95.9928),("Denver",39.7392,-104.9903),("St. Louis",38.6270,-90.1994),("Detroit",42.3314,-83.0458),("Cleveland",41.4993,-81.6944),("Buffalo",42.8864,-78.8784),("New York",40.7128,-74.0060),("Boston",42.3601,-71.0589),("Philadelphia",39.9526,-75.1652),("Washington",38.9072,-77.0369),("Norfolk",36.8508,-76.2859),("Charlotte",35.2271,-80.8431),("Atlanta",33.7490,-84.3880),("Jacksonville",30.3322,-81.6557),("Miami",25.7617,-80.1918),("Tampa",27.9506,-82.4572),("New Orleans",29.9511,-90.0715),("Houston",29.7604,-95.3698),("Los Angeles",34.0522,-118.2437),("San Diego",32.7157,-117.1611),("San Francisco",37.7749,-122.4194),("Seattle",47.6062,-122.3321),("Portland",45.5152,-122.6784)]
MAJOR_CITIES={"Minneapolis","Chicago","Kansas City","Denver","St. Louis","Detroit","Cleveland","New York","Boston","Washington","Atlanta","Houston","Los Angeles","San Francisco","Seattle"}

def cities_for_region(region,max_labels=None):
    c=[(n,lat,lon) for n,lat,lon in US_CITIES if region.south<=lat<=region.north and region.west<=lon<=region.east]
    if max(float(region.east-region.west),float(region.north-region.south))>20:
        c=[x for x in c if x[0] in MAJOR_CITIES]
    return c[:int(max_labels)] if max_labels and len(c)>max_labels else c

def add_city_labels(fig,region,font_size=10):
    import plotly.graph_objects as go
    c=cities_for_region(region)
    if c:
        fig.add_trace(go.Scattergeo(lon=[x[2] for x in c],lat=[x[1] for x in c],text=[x[0] for x in c],mode="markers+text",textposition="top center",name="Cities",marker=dict(size=4,color="rgba(30,30,30,.75)"),textfont=dict(size=font_size,color="rgba(30,30,30,.85)"),hovertemplate="%{text}<extra></extra>",showlegend=False))
    return fig

def add_plotly_cities(fig,region,font_size=10,row=None,col=None):
    import plotly.graph_objects as go
    c=cities_for_region(region)
    if not c: return fig
    trace=go.Scatter(x=[x[2] for x in c],y=[x[1] for x in c],text=[x[0] for x in c],mode="markers+text",textposition="top center",name="Cities",marker=dict(size=5,color="rgba(25,25,25,.80)",line=dict(width=.5,color="white")),textfont=dict(size=font_size,color="rgba(25,25,25,.92)"),hovertemplate="%{text}<br>%{y:.2f}, %{x:.2f}<extra></extra>",showlegend=False)
    fig.add_trace(trace,row=row,col=col) if row is not None and col is not None else fig.add_trace(trace)
    return fig

def add_mpl_cities(ax,region,font_size=8):
    import cartopy.crs as ccrs
    c=cities_for_region(region)
    if not c: return ax
    ax.scatter([x[2] for x in c],[x[1] for x in c],s=9,c="black",edgecolors="white",linewidths=.35,transform=ccrs.PlateCarree(),zorder=20)
    for name,lat,lon in c:
        ax.text(lon,lat+.08,name,fontsize=font_size,ha="center",va="bottom",color="black",transform=ccrs.PlateCarree(),zorder=21,bbox=dict(facecolor="white",edgecolor="none",alpha=.55,pad=.3))
    return ax

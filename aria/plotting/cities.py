from __future__ import annotations
import pandas as pd
CITIES=[
("Minneapolis",44.9778,-93.2650,1),("St. Paul",44.9537,-93.0900,1),("Chicago",41.8781,-87.6298,1),("Milwaukee",43.0389,-87.9065,1),("Des Moines",41.5868,-93.6250,1),("Omaha",41.2565,-95.9345,1),("Sioux Falls",43.5446,-96.7311,1),("Fargo",46.8772,-96.7898,1),("Madison",43.0731,-89.4012,1),("Green Bay",44.5133,-88.0133,1),
("Duluth",46.7867,-92.1005,2),("Rochester",44.0121,-92.4802,2),("Grand Forks",47.9253,-97.0329,2),("Lincoln",40.8136,-96.7026,2),("La Crosse",43.8138,-91.2519,2),("Aberdeen",45.4647,-98.4865,2),("Rapid City",44.0805,-103.2310,2),("Eau Claire",44.8113,-91.4985,2),("Cedar Rapids",41.9779,-91.6656,2),("Davenport",41.5236,-90.5776,2)]
def city_dataframe(region,density="Major"):
    level={"Major":1,"Regional":2,"Dense":3}.get(density,1)
    df=pd.DataFrame(CITIES,columns=["name","latitude","longitude","level"]); df=df[df.level<=level]
    return df[(df.longitude>=region.west)&(df.longitude<=region.east)&(df.latitude>=region.south)&(df.latitude<=region.north)]
def add_plotly_cities(fig,region,density="Major",color="white"):
    import plotly.graph_objects as go
    df=city_dataframe(region,density)
    fig.add_trace(go.Scatter(x=df.longitude,y=df.latitude,mode="markers+text",text=df.name,textposition="top center",
        marker=dict(size=4,color=color,line=dict(width=.5,color="black")),textfont=dict(size=10,color=color),hovertemplate="%{text}<extra></extra>",showlegend=False))
    return fig

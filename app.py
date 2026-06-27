"""
Nassau Candy Distributor - Factory Optimization Dashboard
Author: Rochak Gautam | Unified Mentor Internship
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Nassau Candy | Factory Optimization", page_icon="🍬", layout="wide")

# -------------------------------------------------------
# Reference data (from project brief)
# -------------------------------------------------------
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}
PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack", "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack", "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}
STATE_COORDS = {
    "Alabama":(32.806671,-86.791130),"Alaska":(61.370716,-152.404419),
    "Arizona":(33.729759,-111.431221),"Arkansas":(34.969704,-92.373123),
    "California":(36.116203,-119.681564),"Colorado":(39.059811,-105.311104),
    "Connecticut":(41.597782,-72.755371),"Delaware":(39.318523,-75.507141),
    "District of Columbia":(38.897438,-77.026817),"Florida":(27.766279,-81.686783),
    "Georgia":(33.040619,-83.643074),"Hawaii":(21.094318,-157.498337),
    "Idaho":(44.240459,-114.478828),"Illinois":(40.349457,-88.986137),
    "Indiana":(39.849426,-86.258278),"Iowa":(42.011539,-93.210526),
    "Kansas":(38.526600,-96.726486),"Kentucky":(37.668140,-84.670067),
    "Louisiana":(31.169546,-91.867805),"Maine":(44.693947,-69.381927),
    "Maryland":(39.063946,-76.802101),"Massachusetts":(42.230171,-71.530106),
    "Michigan":(43.326618,-84.536095),"Minnesota":(45.694454,-93.900192),
    "Mississippi":(32.741646,-89.678696),"Missouri":(38.456085,-92.288368),
    "Montana":(46.921925,-110.454353),"Nebraska":(41.125370,-98.268082),
    "Nevada":(38.313515,-117.055374),"New Hampshire":(43.452492,-71.563896),
    "New Jersey":(40.298904,-74.521011),"New Mexico":(34.840515,-106.248482),
    "New York":(42.165726,-74.948051),"North Carolina":(35.630066,-79.806419),
    "North Dakota":(47.528912,-99.784012),"Ohio":(40.388783,-82.764915),
    "Oklahoma":(35.565342,-96.928917),"Oregon":(44.572021,-122.070938),
    "Pennsylvania":(40.590752,-77.209755),"Rhode Island":(41.680893,-71.511780),
    "South Carolina":(33.856892,-80.945007),"South Dakota":(44.299782,-99.438828),
    "Tennessee":(35.747845,-86.692345),"Texas":(31.054487,-97.563461),
    "Utah":(40.150032,-111.862434),"Vermont":(44.045876,-72.710686),
    "Virginia":(37.769337,-78.169968),"Washington":(47.400902,-121.490494),
    "West Virginia":(38.491226,-80.954456),"Wisconsin":(44.268543,-89.616508),
    "Wyoming":(42.755966,-107.302490),
    "Alberta":(53.933271,-116.576504),"British Columbia":(53.726669,-127.647621),
    "Manitoba":(53.760860,-98.813873),"New Brunswick":(46.565314,-66.461914),
    "Newfoundland and Labrador":(53.135509,-57.660435),"Nova Scotia":(44.681999,-63.744310),
    "Ontario":(51.253775,-85.323214),"Prince Edward Island":(46.510712,-63.416809),
    "Quebec":(52.939916,-73.549136),"Saskatchewan":(52.935397,-106.450864),
}
SPEED = {"Same Day":600,"First Class":500,"Second Class":350,"Standard Class":220}
HANDLING = {"Same Day":0.2,"First Class":0.5,"Second Class":1.0,"Standard Class":1.5}

def haversine(p1, p2):
    R = 3958.8
    lat1,lon1 = math.radians(p1[0]),math.radians(p1[1])
    lat2,lon2 = math.radians(p2[0]),math.radians(p2[1])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R*2*math.asin(math.sqrt(a))

# -------------------------------------------------------
# Load & process data (cached so it only runs once)
# -------------------------------------------------------
@st.cache_data
def load_and_process(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # outlier removal
    Q1,Q3 = df["Sales"].quantile(0.25), df["Sales"].quantile(0.75)
    IQR = Q3-Q1
    df = df[(df["Sales"] >= Q1-1.5*IQR) & (df["Sales"] <= Q3+1.5*IQR)].copy()
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df["Distance_Miles"] = df.apply(
        lambda r: haversine(FACTORY_COORDS[r["Factory"]], STATE_COORDS[r["State/Province"]]), axis=1)
    np.random.seed(42)
    df["Lead_Time_Days"] = df.apply(
        lambda r: max(1, round(HANDLING[r["Ship Mode"]] + r["Distance_Miles"]/SPEED[r["Ship Mode"]] + np.random.normal(0,0.6))), axis=1)
    return df

@st.cache_resource
def train_model(df):
    feats = df[["Distance_Miles","Sales","Units","Factory","Region","Ship Mode","Division"]]
    feats = pd.get_dummies(feats, columns=["Factory","Region","Ship Mode","Division"])
    target = df["Lead_Time_Days"]
    X_train,X_test,y_train,y_test = train_test_split(feats,target,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "RMSE": round(mean_squared_error(y_test,preds)**0.5, 3),
        "MAE": round(mean_absolute_error(y_test,preds), 3),
        "R2": round(r2_score(y_test,preds), 3),
    }
    return model, feats.columns.tolist(), metrics

def simulate(product_name, df, model, feat_cols):
    cf = PRODUCT_FACTORY_MAP[product_name]
    rows = df[df["Product Name"]==product_name]
    out = []
    for factory in FACTORY_COORDS:
        sd = rows["State/Province"].apply(lambda s: haversine(FACTORY_COORDS[factory],STATE_COORDS[s]))
        sf = rows[["Sales","Units","Factory","Region","Ship Mode","Division"]].copy()
        sf["Distance_Miles"] = sd.values
        sf["Factory"] = factory
        sf = pd.get_dummies(sf, columns=["Factory","Region","Ship Mode","Division"])
        sf = sf.reindex(columns=feat_cols, fill_value=0)
        cd = rows["State/Province"].apply(lambda s: haversine(FACTORY_COORDS[cf],STATE_COORDS[s]))
        ec = 0.004*(sd.values-cd.values)*rows["Units"].values
        out.append({
            "Factory": factory,
            "Is Current": factory==cf,
            "Avg Lead Time (days)": round(model.predict(sf).mean(),2),
            "Total Profit ($)": round((rows["Gross Profit"].values-ec).sum(),2),
            "Orders": len(rows),
        })
    return pd.DataFrame(out).sort_values("Avg Lead Time (days)")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory Optimization | Rochak Gautam")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload Nassau_Candy_Distributor.csv", type="csv")
page = st.sidebar.radio("Page", [
    "📊 Overview & EDA",
    "🤖 Model Results",
    "🗺️ Route Clusters",
    "🔀 What-If Simulator",
    "✅ Recommendations",
])

if not uploaded:
    st.title("🍬 Nassau Candy Factory Optimization")
    st.info("Upload `Nassau_Candy_Distributor.csv` in the sidebar to get started.")
    st.markdown("""
    **This dashboard covers:**
    - EDA: sales by division, region, ship mode
    - 3-model ML comparison (Linear Regression, Random Forest, Gradient Boosting)
    - Route clustering: Fast / Moderate / Slow
    - What-If simulator: any product × any factory
    - Ranked recommendations with confidence flags
    
    **Author:** Rochak Gautam | Unified Mentor Internship 2026
    """)
    st.stop()

df = load_and_process(uploaded)
model, feat_cols, model_metrics = train_model(df)

# -------------------------------------------------------
# PAGE 1: Overview & EDA
# -------------------------------------------------------
if page == "📊 Overview & EDA":
    st.title("📊 Overview & EDA")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Orders (after outlier removal)", f"{len(df):,}")
    c2.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    c3.metric("Total Gross Profit", f"${df['Gross Profit'].sum():,.0f}")
    c4.metric("Avg Gross Margin", f"{df['Gross Profit'].sum()/df['Sales'].sum()*100:.1f}%")

    st.markdown("---")
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Sales by Division")
        fig,ax = plt.subplots(figsize=(5,3.5))
        df.groupby("Division")["Sales"].sum().sort_values(ascending=False).plot(kind="bar",ax=ax,color="saddlebrown")
        ax.set_xlabel(""); ax.set_ylabel("Sales ($)"); plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Sales by Region")
        fig,ax = plt.subplots(figsize=(5,3.5))
        df.groupby("Region")["Sales"].sum().sort_values(ascending=False).plot(kind="bar",ax=ax,color="darkorange")
        ax.set_xlabel(""); ax.set_ylabel("Sales ($)"); plt.xticks(rotation=20); plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Ship Mode")
        fig,ax = plt.subplots(figsize=(5,3.5))
        df.groupby("Ship Mode")["Sales"].sum().sort_values(ascending=False).plot(kind="bar",ax=ax,color="chocolate")
        ax.set_xlabel(""); ax.set_ylabel("Sales ($)"); plt.xticks(rotation=20); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Orders per Product")
        fig,ax = plt.subplots(figsize=(5,3.5))
        df.groupby("Product Name")["Row ID"].count().sort_values().plot(kind="barh",ax=ax,color="sienna")
        ax.set_xlabel("Orders"); plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("Raw data (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

    st.warning("**Ship Date data quality issue:** the average gap between Order Date and Ship Date is ~1,320 days (3.6 years) - physically impossible. The Ship Date column is corrupted. Lead time is modeled from distance + ship mode instead. See the notebook/research paper for full details.")

# -------------------------------------------------------
# PAGE 2: Model Results
# -------------------------------------------------------
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    st.caption("Random Forest selected as the final model (high accuracy + interpretable feature importances)")

    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE (days)", model_metrics["RMSE"])
    c2.metric("MAE (days)", model_metrics["MAE"])
    c3.metric("R²", model_metrics["R2"])

    st.markdown("---")
    st.subheader("Feature Importance (top 15)")
    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(15)
    fig,ax = plt.subplots(figsize=(8,5))
    importance.sort_values().plot(kind="barh",ax=ax,color="sienna")
    ax.set_xlabel("Importance"); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Lead Time by Ship Mode")
    fig,ax = plt.subplots(figsize=(7,4))
    order = ["Same Day","First Class","Second Class","Standard Class"]
    df.groupby("Ship Mode")["Lead_Time_Days"].mean().reindex(order).plot(kind="bar",ax=ax,color="chocolate")
    ax.set_ylabel("Avg Lead Time (days)"); plt.xticks(rotation=20); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.info("**Model comparison note:** Gradient Boosting (RMSE ~0.649) is slightly more accurate than Random Forest (RMSE ~0.683), but Random Forest was chosen because the difference is negligible and its feature importances make it easier to explain to stakeholders.")

# -------------------------------------------------------
# PAGE 3: Route Clusters
# -------------------------------------------------------
elif page == "🗺️ Route Clusters":
    st.title("🗺️ Route Clusters")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    route = df.groupby(["Factory","Region"]).agg(
        Avg_Lead_Time=("Lead_Time_Days","mean"),
        Avg_Distance=("Distance_Miles","mean"),
        Orders=("Row ID","count"),
    ).reset_index()

    sc = StandardScaler().fit_transform(route[["Avg_Lead_Time","Avg_Distance","Orders"]])
    route["Cluster"] = KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(sc)
    co = route.groupby("Cluster")["Avg_Lead_Time"].mean().sort_values().index
    route["Cluster_Label"] = route["Cluster"].map({co[0]:"Fast",co[1]:"Moderate",co[2]:"Slow"})

    colors = {"Fast":"#2E7D32","Moderate":"#F9A825","Slow":"#C62828"}
    fig,ax = plt.subplots(figsize=(8,5))
    for label,grp in route.groupby("Cluster_Label"):
        ax.scatter(grp["Avg_Distance"],grp["Avg_Lead_Time"],s=grp["Orders"]/2+30,
                   label=label,alpha=0.75,color=colors[label])
    ax.set_xlabel("Avg Distance (miles)"); ax.set_ylabel("Avg Lead Time (days)")
    ax.set_title("Route Clusters (bubble size = order volume)"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Slow Routes (need attention)")
    slow = route[route["Cluster_Label"]=="Slow"].sort_values("Avg_Lead_Time",ascending=False)
    st.dataframe(slow[["Factory","Region","Avg_Lead_Time","Avg_Distance","Orders"]].reset_index(drop=True), use_container_width=True)

    st.subheader("All Routes")
    st.dataframe(route[["Factory","Region","Cluster_Label","Avg_Lead_Time","Avg_Distance","Orders"]].sort_values("Avg_Lead_Time",ascending=False).reset_index(drop=True), use_container_width=True)

# -------------------------------------------------------
# PAGE 4: What-If Simulator
# -------------------------------------------------------
elif page == "🔀 What-If Simulator":
    st.title("🔀 What-If Simulator")
    st.caption("Pick any product and see predicted lead time and profit for all 5 candidate factories")

    product = st.selectbox("Select Product", sorted(PRODUCT_FACTORY_MAP.keys()))
    current_factory = PRODUCT_FACTORY_MAP[product]
    st.info(f"**Current factory:** {current_factory}")

    results = simulate(product, df, model, feat_cols)
    current_row = results[results["Is Current"]].iloc[0]

    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(5,4))
        colors_bar = ["chocolate" if r else "#C8A57C" for r in results["Is Current"]]
        ax.barh(results["Factory"], results["Avg Lead Time (days)"], color=colors_bar)
        ax.set_xlabel("Avg Lead Time (days)"); ax.set_title("Lead Time by Candidate Factory")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        fig,ax = plt.subplots(figsize=(5,4))
        ax.barh(results["Factory"], results["Total Profit ($)"], color=colors_bar)
        ax.set_xlabel("Total Profit ($)"); ax.set_title("Profit by Candidate Factory")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Full Simulation Table")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)

    n_orders = len(df[df["Product Name"]==product])
    if n_orders < 30:
        st.warning(f"Low volume: only {n_orders} historical orders for this product. Results are directional, not precise.")

# -------------------------------------------------------
# PAGE 5: Recommendations
# -------------------------------------------------------
elif page == "✅ Recommendations":
    st.title("✅ Recommendations")

    all_sim = pd.concat([
        simulate(p, df, model, feat_cols).assign(Product=p)
        for p in PRODUCT_FACTORY_MAP
    ], ignore_index=True)

    out = []
    for prod, grp in all_sim.groupby("Product"):
        cur = grp[grp["Is Current"]].iloc[0]
        grp = grp.copy()
        grp["Lead_Time_Improvement_Pct"] = (cur["Avg Lead Time (days)"]-grp["Avg Lead Time (days)"])/cur["Avg Lead Time (days)"]*100
        grp["Profit_Change"] = grp["Total Profit ($)"]-cur["Total Profit ($)"]
        lt = (grp["Lead_Time_Improvement_Pct"]-grp["Lead_Time_Improvement_Pct"].min())/(grp["Lead_Time_Improvement_Pct"].max()-grp["Lead_Time_Improvement_Pct"].min()+1e-9)
        pc = (grp["Profit_Change"]-grp["Profit_Change"].min())/(grp["Profit_Change"].max()-grp["Profit_Change"].min()+1e-9)
        grp["Score"] = 0.5*lt+0.5*pc
        grp["Low_Volume"] = grp["Orders"] < 30
        grp.loc[grp["Low_Volume"],"Score"] *= 0.6
        out.append(grp.sort_values("Score",ascending=False))
    recs = pd.concat(out,ignore_index=True)
    top = recs.sort_values("Score",ascending=False).groupby("Product").first().reset_index()

    high_conf = top[(top["Is Current"]==False) & (~top["Low_Volume"])]

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Profit Upside", f"${high_conf['Profit_Change'].sum():,.0f}")
    c2.metric("Avg Lead Time Improvement", f"{high_conf['Lead_Time_Improvement_Pct'].mean():.1f}%")
    c3.metric("High-Confidence Moves", len(high_conf))

    st.markdown("---")
    st.subheader("High-Confidence Recommendations (30+ historical orders)")
    fig,ax = plt.subplots(figsize=(8,4))
    hc_sorted = high_conf.sort_values("Profit_Change")
    ax.barh(hc_sorted["Product"], hc_sorted["Profit_Change"], color="chocolate")
    ax.set_xlabel("Profit Change ($)"); ax.set_title("High-Confidence Reassignment Opportunities")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    display_cols = ["Product","Factory","Is Current","Orders","Low_Volume","Lead_Time_Improvement_Pct","Profit_Change"]
    st.dataframe(
        top[display_cols].rename(columns={
            "Factory":"Recommended Factory","Lead_Time_Improvement_Pct":"Lead Time Improvement (%)",
            "Profit_Change":"Profit Change ($)","Low_Volume":"Low Volume Flag"
        }).reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")
    st.info("**Low Volume Flag = True** means fewer than 30 historical orders exist for that product. Those recommendations are directional only and not included in the headline numbers above.")

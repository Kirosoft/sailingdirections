import folium
from config import ES_FEATURES_INDEX, es

def get_all_features(es, index_name):
    """Fetch all features from the features index."""
    body = {"query": {"match_all": {}}}
    res = es.search(index=index_name, body=body, size=10000)  # adjust size as needed
    return [
        {
            "feature_id": h["_source"]["feature_id"],
            "name": h["_source"]["name"],
            "location": h["_source"]["location"],
            "section_id": h["_source"]["section_id"]
        }
        for h in res["hits"]["hits"]
    ]

def add_features_to_map(m, features):
    for feat in features:
        loc = feat["location"]
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"{feat['name']}\n({loc['lat']:.4f}, {loc['lon']:.4f})",
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)

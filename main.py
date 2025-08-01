def get_passage_plan_bucks_to_somes():
    """
    Returns a passage plan (list of waypoints) for a ship from Bucks Harbour to Somes Sound.
    Bucks Harbour: 44.3367, -68.7425
    Somes Sound: 44.3333, -68.3117
    """
    # Waypoints chosen to avoid land and follow a safe channel
    return [
        (44.3367, -68.7425),  # Bucks Harbour
        (44.3400, -68.7000),  # Off Pond Island
        (44.3400, -68.6500),  # Off Black Island
        (44.2700, -68.5000),  # Off Swans Island (north of)
        (44.3000, -68.3500),  # Off Bear Island
        (44.3333, -68.3117)   # Somes Sound
    ]

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
import folium
from ingest import parse_and_chunk
from passage import refine_passage_geospatial
from search import ensure_features_index, ensure_index, index_sections, semantic_search, geo_search_dms, lexical_search, hybrid_search


from config import ES_INDEX_NAME, ES_FEATURES_INDEX, RE_INDEX, es



# ——— Main —————————————————————————————————————————————
if __name__=="__main__":
    
    if RE_INDEX:
        with open("e-NP68_17_2021-chapter2.md",encoding="utf-8") as f: 
            raw=f.read()
            secs=parse_and_chunk(raw)
            ensure_index(es,ES_INDEX_NAME)
            ensure_features_index(es,ES_FEATURES_INDEX)
            index_sections(es,secs,ES_INDEX_NAME,ES_FEATURES_INDEX)
            print(f"Indexed {len(secs)} sections with features into {ES_INDEX_NAME} and {ES_FEATURES_INDEX}")
    else:
        print(f"Skipping re-indexing, using existing indices {ES_INDEX_NAME} and {ES_FEATURES_INDEX}")

    print("\n-- Semantic Search --")
    for r in semantic_search(es,ES_INDEX_NAME,"Bar Island",k=3): print(r)
    print("\n-- Geo Search (DMS) --")
    for r in geo_search_dms(es,"442390N 681240W","1m"): print(r)
    print("\n-- Lexical Search --")
    for r in lexical_search(es,ES_INDEX_NAME,"Bar Island"): print(r)
    print("\n-- Hybrid Search --")
    for r in hybrid_search(es,ES_INDEX_NAME,"Bar Island",alpha=0.7,k=5): print(r)


    # A ship passage from Bucks Harbour to Somes Sound (with safe waypoints)
    ship_passage_latlon = get_passage_plan_bucks_to_somes()

    # Set the desired distance between points in kilometers.
    distance_between_points_km = 10.0

    print("Generating refined passage for geospatial coordinates...")
    # Generate the high-resolution path using the geospatial function.
    fine_grained_passage_geo = refine_passage_geospatial(ship_passage_latlon, distance_between_points_km)

    # --- Create Map with Folium ---
    print("Creating map...")
    # Center the map on the average of the waypoints.
    avg_lat = sum(p[0] for p in ship_passage_latlon) / len(ship_passage_latlon)
    avg_lon = sum(p[1] for p in ship_passage_latlon) / len(ship_passage_latlon)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9, tiles="OpenStreetMap")

    # Add original waypoints to the map as large blue markers
    for point in ship_passage_latlon:
        folium.Marker(
            location=point,
            popup=f"Original Waypoint\n({point[0]:.4f}, {point[1]:.4f})",
            icon=folium.Icon(color='blue', icon='flag')
        ).add_to(m)

    # Add refined path to the map as a red polyline
    folium.PolyLine(
        locations=fine_grained_passage_geo,
        color='red',
        weight=3,
        opacity=0.8,
        tooltip="Refined Passage"
    ).add_to(m)


    # Add small markers for the newly interpolated points
    for point in fine_grained_passage_geo:
        if point not in ship_passage_latlon:
            folium.CircleMarker(
                location=point,
                radius=3,
                color='red',
                fill=True,
                fill_color='darkred',
                fill_opacity=0.7,
                popup=f"Interpolated Point\n({point[0]:.4f}, {point[1]:.4f})"
            ).add_to(m)

    # --- Add features from Elasticsearch ---
    from feature_map import get_all_features, add_features_to_map
    print("Loading features from Elasticsearch and adding to map...")
    features = get_all_features(es, ES_FEATURES_INDEX)
    add_features_to_map(m, features)

    # Save the map to an HTML file
    output_filename = "ship_passage_map.html"
    m.save(output_filename)

    print("-" * 30)
    print(f"Map has been generated and saved to '{output_filename}'")
    print(f"Original waypoints: {len(ship_passage_latlon)}")
    print(f"Refined waypoints: {len(fine_grained_passage_geo)}")
    print("-" * 30)
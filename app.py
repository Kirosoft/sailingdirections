from flask import Flask, render_template, request, jsonify
import folium

from main import get_passage_plan_bucks_to_somes, refine_passage_geospatial
from feature_map import get_all_features, add_features_to_map
from config import ES_FEATURES_INDEX, es

app = Flask(__name__)

@app.route('/')
def index():
    # Default route: Bucks Harbour to Somes Sound
    waypoints = get_passage_plan_bucks_to_somes()
    refined = refine_passage_geospatial(waypoints, 10.0)
    # Generate map HTML
    avg_lat = sum(p[0] for p in waypoints) / len(waypoints)
    avg_lon = sum(p[1] for p in waypoints) / len(waypoints)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9, tiles="OpenStreetMap")
    folium.PolyLine(locations=refined, color='red', weight=3, opacity=0.8, tooltip="Refined Passage").add_to(m)
    for point in waypoints:
        folium.Marker(location=point, popup=f"Waypoint\n({point[0]:.4f}, {point[1]:.4f})", icon=folium.Icon(color='blue', icon='flag')).add_to(m)
    # Load features from Elasticsearch and add to map
    features = get_all_features(es, ES_FEATURES_INDEX)
    add_features_to_map(m, features)
    # Inject JS for ALT+click event in the map iframe
    from folium import MacroElement
    from jinja2 import Template
    class AltClickJS(MacroElement):
        _template = Template(r"""
{% macro script(this, kwargs) %}
if (typeof window.L !== 'undefined' && {{this._parent.get_name()}}) {
    {{this._parent.get_name()}}.on('click', function(e) {
        if (e.originalEvent && e.originalEvent.altKey) {
            window.parent.postMessage({lat: e.latlng.lat, lng: e.latlng.lng, alt: true}, '*');
        }
    });
}
{% endmacro %}
""")
    m.get_root().add_child(AltClickJS())
    map_html = m._repr_html_()
    return render_template('index.html', map_html=map_html, sailing_directions="Sailing directions will appear here.")

@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.json
    waypoints = data.get('waypoints', [])
    if not waypoints:
        return jsonify({'error': 'No waypoints provided'}), 400
    refined = refine_passage_geospatial(waypoints, 10.0)
    # For demo, just return the refined route
    return jsonify({'refined': refined})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)

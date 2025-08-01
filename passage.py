import math
from typing import List, Tuple

def refine_passage(
    waypoints: List[Tuple[float, float]],
    interpolation_distance: float
) -> List[Tuple[float, float]]:
    """
    Refines a passage plan for 2D CARTESIAN coordinates.

    This function is suitable for abstract grids or flat-plane projections.
    For real-world latitude/longitude coordinates, use the
    `refine_passage_geospatial` function instead.

    Args:
        waypoints: A list of (x, y) tuples.
        interpolation_distance: The desired distance between points.

    Returns:
        A new, more detailed list of waypoints.
    """
    if interpolation_distance <= 0:
        raise ValueError("Interpolation distance must be a positive number.")

    if not waypoints or len(waypoints) < 2:
        return waypoints

    refined_path = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        segment_length = math.hypot(dx, dy)

        if segment_length <= interpolation_distance:
            if end_point not in refined_path:
                 refined_path.append(end_point)
            continue

        num_new_points = int(segment_length / interpolation_distance)
        unit_dx = dx / segment_length
        unit_dy = dy / segment_length

        for j in range(1, num_new_points + 1):
            step_distance = j * interpolation_distance
            new_x = start_point[0] + step_distance * unit_dx
            new_y = start_point[1] + step_distance * unit_dy
            refined_path.append((new_x, new_y))

        if end_point not in refined_path:
            refined_path.append(end_point)

    return refined_path

# --- Geospatial Functions for Latitude/Longitude ---

def haversine_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates the great-circle distance between two points on Earth."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_initial_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates the initial bearing from point 1 to point 2."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
    
    initial_bearing = math.atan2(x, y)
    return math.degrees(initial_bearing)

def find_destination_point(start_point: Tuple[float, float], bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    """Finds a destination point given a start point, bearing, and distance."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = math.radians(start_point[0]), math.radians(start_point[1])
    bearing = math.radians(bearing_deg)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
    
    return (math.degrees(lat2), math.degrees(lon2))

def refine_passage_geospatial(
    waypoints: List[Tuple[float, float]],
    interpolation_distance_km: float
) -> List[Tuple[float, float]]:
    """
    Refines a passage plan using GEOSPATIAL coordinates (lat/lon).
    
    This function correctly calculates distances and interpolates points along
    great-circle paths on the Earth's surface.
    """
    if interpolation_distance_km <= 0:
        raise ValueError("Interpolation distance must be a positive number.")
    if not waypoints or len(waypoints) < 2:
        return waypoints

    refined_path = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]

        segment_length = haversine_distance(start_point, end_point)
        if segment_length <= interpolation_distance_km:
            if end_point not in refined_path:
                refined_path.append(end_point)
            continue

        bearing = calculate_initial_bearing(start_point, end_point)
        num_new_points = int(segment_length / interpolation_distance_km)

        for j in range(1, num_new_points + 1):
            step_distance = j * interpolation_distance_km
            new_point = find_destination_point(start_point, bearing, step_distance)
            refined_path.append(new_point)

        if end_point not in refined_path:
            refined_path.append(end_point)
            
    return refined_path
import numpy as np
from sklearn.neighbors import BallTree
from collections import defaultdict, deque
import math

def retrieve_restaurants_and_ids(df):
    restaurant_ids = df.iloc[:, 0].values
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values
    restaurants = np.vstack((latitudes, longitudes)).T

    return restaurants, restaurant_ids


def create_id_mappings(restaurant_ids):
    id_to_index = {rid: idx for idx, rid in enumerate(restaurant_ids)}
    index_to_id = {idx: rid for idx, rid in enumerate(restaurant_ids)}
    return id_to_index, index_to_id


def generate_candidate_circles(restaurants, index_to_id, max_neighbors=17):
    EARTH_RADIUS_KM = 6371.0
    restaurants_rad = np.radians(restaurants)
    ball_tree = BallTree(restaurants_rad, metric='haversine')

    candidate_circles = []

    for i, point in enumerate(restaurants_rad):
        distances, indices = ball_tree.query(point.reshape(1, -1), k=max_neighbors)
        distances_km = distances[0] * EARTH_RADIUS_KM
        radius = distances_km[-1]

        # Account for dataset error for special building types such as airports that plot all points to a single coordinate
        if radius == 0:
            radius = 0.05

        # Include the center point in the covered set to ensure self-coverage
        covered_ids = set(index_to_id[idx] for idx in indices[0])
        covered_ids.add(index_to_id[i])

        circle = {
            'center': restaurants[i],
            'radius': radius,  # Radius in kilometers
            'points_covered': covered_ids
        }
        candidate_circles.append(circle)

    return candidate_circles


def greedy_set_cover_bucket_queue(candidate_circles, restaurant_ids, prioritize = False):
    # Greedy Set Cover using a linear-time bucket queue solution.
    uncovered_points = set(restaurant_ids)
    selected_circles = []

    # Step 1: Create a mapping from each element to the sets that contain it
    element_to_sets = defaultdict(set)
    for i, circle in enumerate(candidate_circles):
        for point in circle['points_covered']:
            element_to_sets[point].add(i)

    # Step 2: Initialize the bucket queue and track current uncovered count for each circle
    max_uncovered_count = 0
    set_map = defaultdict(deque)  # Each bucket contains a deque of set indices
    current_uncovered_count = {}  # Track current uncovered count for each circle

    for i, circle in enumerate(candidate_circles):
        uncovered_count = len(circle['points_covered'])
        if prioritize:
            set_map[uncovered_count].append((i, -circle['radius']))
        else:
            set_map[uncovered_count].append(i)

        current_uncovered_count[i] = uncovered_count
        max_uncovered_count = max(max_uncovered_count, uncovered_count)

    # Step 3: Greedy selection loop
    while uncovered_points:
        # Find the set with the maximum uncovered count
        while max_uncovered_count > 0 and not set_map[max_uncovered_count]:
            max_uncovered_count -= 1
        if max_uncovered_count == 0:
            break

        # Select a circle from the current max bucket
        if prioritize:
            set_map[max_uncovered_count] = deque(sorted(set_map[max_uncovered_count], key=lambda x: x[1]))
            circle_index, _ = set_map[max_uncovered_count].popleft()
        else:
            circle_index = set_map[max_uncovered_count].popleft()
        circle = candidate_circles[circle_index]

        # Calculate the newly covered points
        newly_covered = circle['points_covered'].intersection(uncovered_points)

        # Update uncovered points and add the selected circle
        uncovered_points -= newly_covered
        selected_circles.append(circle)

        # Step 4: Update only affected sets
        affected_sets = set()
        for point in newly_covered:
            affected_sets.update(element_to_sets[point])
            element_to_sets[point].clear()  # Clear as point is now covered

        for i in affected_sets:
            if i == circle_index:
                continue

            other_circle = candidate_circles[i]

            # Calculate the new uncovered count for the affected circle
            new_uncovered = other_circle['points_covered'].intersection(uncovered_points)
            new_count = len(new_uncovered)
            old_count = current_uncovered_count[i]

            # Update set_map and current_uncovered_count if the count has changed
            if new_count != old_count:
                if prioritize:
                    set_map[old_count].remove((i, -other_circle['radius']))
                    set_map[new_count].append((i, -other_circle['radius']))
                else:
                    set_map[old_count].remove(i)
                    set_map[new_count].append(i)
                current_uncovered_count[i] = new_count  # Update the tracked count

    return selected_circles


def find_dual_range_space(X, Ranges):
    """Create the dual range space (X⊥, R⊥) based on unique identifiers."""
    X_dual = list(range(len(Ranges)))  # Each element in X_dual represents a range in the original Ranges
    R_dual = {x: set() for x in X}  # Each R_x will contain indices of ranges that include x

    # Populate R_dual with indices of ranges containing each unique identifier in X
    for i, R in enumerate(Ranges):
        for x in R:
            if x in R_dual:
                R_dual[x].add(i)

    # Convert R_dual to a list of sets for easier processing in dual form
    R_dual_list = [R_dual[x] for x in X]

    return X_dual, R_dual_list


def transform_to_hitting_set_problem(restaurant_ids, candidate_circles):
    X = np.array(restaurant_ids)
    Ranges = [circle["points_covered"] for circle in candidate_circles]
    X_dual, R_dual_list = find_dual_range_space(X, Ranges)
    return X, Ranges, X_dual, R_dual_list


def hitting_set_search(X, Ranges, X_dual, R_dual_list):
    powers_of_2 = [2 ** i for i in range(int(math.log2(len(X))) + 1)]
    lower_bound = 0
    upper_bound = len(powers_of_2) - 1
    optimal_set_cover = None

    while lower_bound <= upper_bound:
        mid_point = (lower_bound + upper_bound) // 2
        t = powers_of_2[mid_point]
        set_cover = hitting_set_in_rounds(X_dual, R_dual_list, X, Ranges, t)

        if set_cover:
            optimal_set_cover = set_cover
            upper_bound = mid_point - 1
        else:
            lower_bound = mid_point + 1
    
    return optimal_set_cover


def hitting_set_in_rounds(X, ranges, restaurant_X, restaurant_ranges, t):
        # Initialize weights
        hitting_set = None
        weights = {x: 1 for x in X}
        found_hitting_set = False
        max_rounds = 2 * math.log2(len(X) / t) + 1  # Max rounds based on t
        round_counter = 0
        total_weight = sum(weights.values())  # Sum of all weights in X

        # Iterate rounds for the current value of t
        while not found_hitting_set and round_counter < max_rounds:
            round_counter += 1
            heavy_threshold = (1 / (2 * t)) * total_weight
            doubling_steps = 0  # Reset for each round
            restart_round = False

            for R in ranges:
                # Compute the weight of points in R
                weight_in_R = sum(weights[x] for x in R)

                # Weight-doubling to reach "heavy" threshold
                while weight_in_R < heavy_threshold:
                    added_weight = 0
                    for x in R:
                        added_weight += weights[x]
                        weights[x] *= 2  # Double weights in R
                    weight_in_R += added_weight
                    total_weight += added_weight
                    heavy_threshold = (1 / (2 * t)) * total_weight
                    doubling_steps += 1

                    # Restart round if exceeding doubling steps
                    if doubling_steps >= 2 * t:
                        restart_round = True
                        break  # Restart required

                if restart_round:
                    break  # Restart the round

            if restart_round:
                continue  # Reinitialize round with current weights

            # Hitting set found if all rounds completed without restart
            found_hitting_set = True

            # Construct hitting set by selecting the highest-weighted point from each range
            hitting_set = set()
            covered_ranges = set()  # Track covered ranges
            for i, R in enumerate(ranges): #point = a range index #ranges = set of range indices that include point x
                if i in covered_ranges: #*****
                    continue
                max_weight_point = max(R, key=lambda x: weights[x])
                hitting_set.add(max_weight_point)

                for p in restaurant_ranges[max_weight_point]: #*****
                    covered_ranges.add(np.where(restaurant_X == p)[0][0]) #***** change so map unique identifier to index in X

        return hitting_set  # Return found hitting set
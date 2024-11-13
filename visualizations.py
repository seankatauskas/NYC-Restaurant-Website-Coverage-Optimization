from utils import download_and_unzip
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from shapely.geometry import Polygon

def create_geodataframe(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs='EPSG:4326')
    return gdf

def download_streets_boroughs_shapefiles():
    url_streets = 'https://planninglabs.carto.com/api/v2/sql?format=SHP&filename=dcp_dcm_clipped&q=SELECT%20*%20FROM%20dcp_dcm%20WHERE%20ST_Intersects(the_geom,%20ST_GeomFromGeoJSON(%27{%22type%22:%22Polygon%22,%22coordinates%22:[[[-74.40604194412995,40.94372122495801],[-72.97747416762984,40.94372122495801],[-72.97747416762984,39.96347115796587],[-74.40604194412995,39.96347115796587],[-74.40604194412995,40.94372122495801]]],%22crs%22:{%22type%22:%22name%22,%22properties%22:{%22name%22:%22EPSG:4326%22}}}%27))%20AND%20feat_type%20NOT%20IN%20(%27Pierhead_line%27,%20%27Bulkhead_line%27,%20%27Rail%27)'
    url_boroughs = 'https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile'
    download_and_unzip(url_streets, 'streets')
    download_and_unzip(url_boroughs, 'boroughs')

    return True

def load_borough_shapefiles(folder_path):
    borough_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.shp')]
    boroughs_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(file) for file in borough_files], ignore_index=True))
    return boroughs_gdf

def load_shapefiles():
    street_shapefile = 'streets'  # Replace with actual file path
    borough_folder = 'boroughs'  # Replace with the actual folder path
    nyc_streets = gpd.read_file(street_shapefile)
    nyc_boroughs = load_borough_shapefiles(borough_folder)

    return nyc_streets, nyc_boroughs

# Helper function to plot borough boundaries
def plot_borough_boundaries(ax, nyc_boroughs, color='black', linestyle='-', linewidth=0.5):
    nyc_boroughs.boundary.plot(ax=ax, color=color, linewidth=linewidth, linestyle=linestyle)

# Helper function to set labels and title
def set_plot_labels(ax, title, xlabel='Longitude', ylabel='Latitude'):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Helper function to add colorbar
def add_colorbar(fig, sc, ax, label):
    cbar = fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04, aspect=30)
    cbar.set_alpha(1)

# Helper function to generate grid counts for heatmap
def generate_grid_counts(gdf, grid_size):
    gdf['grid_x'] = (gdf['Longitude'] // grid_size) * grid_size
    gdf['grid_y'] = (gdf['Latitude'] // grid_size) * grid_size
    return gdf.groupby(['grid_x', 'grid_y']).size().reset_index(name='Count')

# General Density and Heat Map Plot
def plot_general_density_and_heatmap(gdf, nyc_boroughs, grid_size=0.004):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)

    # Plot 1: Initial Point Distribution
    plot_borough_boundaries(axes[0], nyc_boroughs)
    gdf.plot(ax=axes[0], color='blue', alpha=0.5, markersize=2)
    set_plot_labels(axes[0], 'Map of Restaurants in NYC')
    
    # Plot 2: Grid-Based Visualization
    grid_counts = generate_grid_counts(gdf, grid_size)
    sc = axes[1].scatter(
        grid_counts['grid_x'], grid_counts['grid_y'],
        c=grid_counts['Count'],
        cmap='plasma',
        s=150, alpha=0.7,
        edgecolor='k', linewidth=0.5
    )
    plot_borough_boundaries(axes[1], nyc_boroughs)
    add_colorbar(fig, sc, axes[1], label='Number of Restaurants')
    set_plot_labels(axes[1], 'Grid-Based Visualization of NYC Restaurants')
    plt.show()


# DBSCAN Clustering Heat Map Plot
def plot_dbscan_clusters(gdf, nyc_boroughs, nyc_streets, eps=0.0001, min_samples=100, grid_size=0.002):
    # Convert latitude and longitude to radians for Haversine distance
    coords = np.radians(gdf[['Latitude', 'Longitude']].values)

    # Apply DBSCAN with Haversine distance
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    gdf['cluster'] = dbscan.fit_predict(coords)

    # Count the number of points in each cluster, excluding noise (-1)
    cluster_counts = gdf['cluster'].value_counts().drop(-1, errors='ignore')
    top_clusters = cluster_counts.head(6).index

    # Create a figure and axes for subplots (2 rows, 3 columns for 6 clusters)
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))  # Increase the figure size
    axes = axes.flatten()  # Flatten to easily iterate over axes

    for i, cluster in enumerate(top_clusters):
        ax = axes[i]

        # Create a copy of the filtered gdf to avoid SettingWithCopyWarning
        cluster_gdf = gdf[gdf['cluster'] == cluster].copy()
        num_restaurants = len(cluster_gdf)  # Calculate the number of restaurants in this cluster

        # Calculate grid cells for each cluster using .loc[] on the copied DataFrame
        cluster_gdf.loc[:, 'grid_x'] = (cluster_gdf['Longitude'] // grid_size) * grid_size
        cluster_gdf.loc[:, 'grid_y'] = (cluster_gdf['Latitude'] // grid_size) * grid_size

        # Group by grid cells and count restaurants in each cell
        grid_counts = cluster_gdf.groupby(['grid_x', 'grid_y']).size().reset_index(name='Count')

        # Create a pivot table for heatmap creation
        pivot_table = grid_counts.pivot(index='grid_y', columns='grid_x', values='Count').fillna(0)

        # Calculate the cluster centroid for borough identification
        center_lat = cluster_gdf['Latitude'].mean()
        center_lon = cluster_gdf['Longitude'].mean()
        borough_name = get_borough(center_lat, center_lon)

        # Calculate the bounding box for the cluster, aligned with the grid size
        grid_half_size = grid_size / 2
        min_lon = ((cluster_gdf['Longitude'].min() // grid_size) * grid_size) - grid_half_size
        max_lon = ((cluster_gdf['Longitude'].max() // grid_size) * grid_size) + grid_half_size
        min_lat = ((cluster_gdf['Latitude'].min() // grid_size) * grid_size) - grid_half_size
        max_lat = ((cluster_gdf['Latitude'].max() // grid_size) * grid_size) + grid_half_size

        # Plot the borough outlines with a lighter color and thinner lines
        nyc_boroughs.boundary.plot(ax=ax, color='lightblue', linewidth=0.8, linestyle='--')

        # Plot the major streets with a more distinct color
        nyc_streets.plot(ax=ax, color='darkgray', linewidth=0.5)

        # Plot the heatmap with increased opacity and a vibrant color palette
        heatmap = ax.pcolormesh(
            pivot_table.columns, pivot_table.index, pivot_table.values,
            cmap='inferno',  # Use 'inferno' colormap for higher contrast
            shading='auto',
            alpha=1  # Make fully opaque
        )

        # Set the axis limits to zoom in on the cluster area
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

        # Set titles and labels, including the borough name
        ax.set_title(f'Cluster {cluster} in {borough_name} (Restaurants: {num_restaurants})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add a colorbar with adjusted size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(heatmap, cax=cax, label='Number of Restaurants')

    # Remove any unused subplot axes (if 6 plots in a 2x3 grid)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust spacing between plots for better visualization
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust vertical and horizontal space
    plt.show()



def get_borough(lat, lon):
    if 40.7 <= lat <= 40.8 and -74.0 <= lon <= -73.9:
        return 'Manhattan'
    elif 40.6 <= lat <= 40.7 and -74.1 <= lon <= -73.8:
        return 'Brooklyn'
    elif 40.7 <= lat <= 40.8 and -73.8 <= lon <= -73.7:
        return 'Queens'
    elif 40.5 <= lat <= 40.6 and -74.2 <= lon <= -74.1:
        return 'Staten Island'
    elif 40.8 <= lat <= 40.9 and -73.9 <= lon <= -73.8:
        return 'Bronx'
    else:
        return 'Unknown'

# Helper function for clipping Voronoi regions
def voronoi_region_within_convex_hull(vor, hull_points):
    regions = []
    vertices = vor.vertices
    hull_polygon = Polygon(hull_points)

    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if -1 in region:
            continue
        region_polygon = Polygon(vertices[region])
        clipped_polygon = region_polygon.intersection(hull_polygon)
        if clipped_polygon.is_valid and not clipped_polygon.is_empty:
            regions.append(clipped_polygon)
    return regions

# Voronoi Diagram Plot
def plot_voronoi_diagram(gdf):
    coords = gdf[['Longitude', 'Latitude']].to_numpy()
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    vor = Voronoi(coords)
    clipped_regions = voronoi_region_within_convex_hull(vor, hull_points)

    fig, ax = plt.subplots(figsize=(15, 15))
    hull_polygon = Polygon(hull_points)
    x, y = hull_polygon.exterior.xy
    ax.plot(x, y, color='green', linestyle='--', label='Convex Hull')

    for region in clipped_regions:
        x, y = region.exterior.xy
        ax.fill(x, y, alpha=0.4, edgecolor='orange', facecolor='lightblue')

    ax.scatter(gdf['Longitude'], gdf['Latitude'], color='blue', s=5, alpha=0.5, label='Restaurants')
    set_plot_labels(ax, 'Voronoi Diagram Bounded by Convex Hull of Restaurants in NYC')
    plt.legend(loc='upper right')
    plt.show()


# Delaunay Triangulation and K-Means Plot
def plot_delaunay_triangulation_and_kmeans(gdf, nyc_boroughs, n_clusters=10):
    coords = gdf[['Longitude', 'Latitude']].values
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)

    tri = Delaunay(coords)
    plot_borough_boundaries(axes[0], nyc_boroughs)
    axes[0].triplot(coords[:, 0], coords[:, 1], tri.simplices, color='red', alpha=0.5)
    axes[0].plot(coords[:, 0], coords[:, 1], 'o', markersize=2, color='blue', alpha=0.5)
    set_plot_labels(axes[0], 'Delaunay Triangulation')

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    plot_borough_boundaries(axes[1], nyc_boroughs)
    axes[1].scatter(coords[:, 0], coords[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5, s=5)
    axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, marker='X')
    set_plot_labels(axes[1], 'K-Means Clustering of Restaurants')
    plt.show()


def plot_restaurant_distribution_with_circles(nyc_boroughs, selected_circles, figsize=(20, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    nyc_boroughs.boundary.plot(ax=ax, color='black', linewidth=0.5, linestyle='-', label='Borough Boundaries')
    set_plot_labels(ax, title='Map of Restaurants in NYC with Candidate Circles', xlabel='Longitude', ylabel='Latitude')

    for idx, circle in enumerate(selected_circles):
        center_lat, center_lon = circle['center']
        radius_km = circle['radius']
        radius_deg = radius_km / 111  # Convert radius from kilometers to degrees
        circle_patch = Circle(
            (center_lon, center_lat),
            radius=radius_deg,
            color='red', alpha=0.3, fill=False, linestyle='-'
        )
        ax.add_patch(circle_patch)

    ax.legend()
    plt.show()
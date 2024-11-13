import pandas as pd
from utils import update_address
from geopy.geocoders import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import csv

# Function to construct the full address
def construct_full_address(row):
    # Construct the address from available fields
    building = row['BUILDING']
    street = row['STREET']
    zipcode = row['ZIPCODE']
    boro = row['BORO']

    # Check if building is '000' and change it to '0'
    if building == '000':
        building = '0'

    # Convert ZIPCODE to string, handle NaN
    if pd.notna(zipcode):
        zipcode = str(int(zipcode))
    else:
        zipcode = ''

    # Construct the address parts list
    address_parts = []

    # Add building and street together if both are present
    if pd.notna(building) and building != '':
        if pd.notna(street) and street != '':
            address_parts.append(f"{building} {street}")
        else:
            address_parts.append(building)  # Only building number
    elif pd.notna(street) and street != '':
        address_parts.append(street)  # Only street if building is missing

    address_parts[0] = update_address(address_parts[0]) #all rows into this function have a street or building

    # Add borough, New York, NY, and zipcode, separated by commas
    if pd.notna(boro) and boro != '' and boro != '0':
        address_parts.append(boro)
    address_parts.append('New York, NY')  # Static part for city/state
    if pd.notna(zipcode) and zipcode != '':
        address_parts.append(zipcode)

    # Join the parts into a single string, separated by commas
    address = ', '.join(address_parts)

    return address


def load_restaurant_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=df.columns[0], keep='first')
    df.reset_index(drop=True, inplace=True)

    # Drop rows where:
    # - (Latitude or longitude are 0 or NaN) and (Both 'BUILDING' and 'STREET' fields are empty or NaN)
    # - 'BORO' is '0'
    df = df[
    ~(
        (((df['Latitude'] == 0) | (df['Latitude'].isna())) |
        ((df['Longitude'] == 0) | (df['Longitude'].isna()))) &
        (df['BUILDING'].isna() | (df['BUILDING'] == '')) &
        (df['STREET'].isna() | (df['STREET'] == ''))
    ) &
    (df['BORO'] != '0')
    ]
    df.reset_index(drop=True, inplace=True)

    return df


def populate_full_address(df):
    df['FULL_ADDRESS'] = df.apply(construct_full_address, axis=1)
    return df


def populate_missing_coordinates(df, api_key):
    tqdm.pandas()

    def geocode_full_address(address):
        location = geocode(address)
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
        
    # Select rows where (latitude or longitude are 0 or NaN)
    df_no_coordinates = df[
    (((df['Latitude'] == 0) | (df['Latitude'].isna())) |
    ((df['Longitude'] == 0) | (df['Longitude'].isna())))
    ].copy()

    print(df_no_coordinates.shape[0])

    # Initialize geopy with GoogleV3 geocoder
    geolocator = GoogleV3(api_key, timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1)
    df_no_coordinates[['Latitude', 'Longitude']] = df_no_coordinates['FULL_ADDRESS'].progress_apply(geocode_full_address)

    # Merge on unique identifier for df_no_coordinates into df
    df.set_index('CAMIS', inplace=True)
    df_no_coordinates.set_index('CAMIS', inplace=True)
    df.update(df_no_coordinates[['Latitude', 'Longitude']])
    df.reset_index(inplace=True)

    return df


def refine_geocode_for_restaurants(df, api_key):
    tqdm.pandas()

    def geocode_combined_name_address(query):
        location = geocode(query)
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
        
    def create_geocode_query(row):
        return f"{row['DBA']} - {row['FULL_ADDRESS']}"
    
    df_shared_coordinates = df[df.duplicated(subset=['Latitude', 'Longitude'], keep=False)].copy()
    df_shared_coordinates['GEOCODE_QUERY'] = df_shared_coordinates.apply(create_geocode_query, axis=1)
    geocoded_coordinates = df_shared_coordinates[['Latitude', 'Longitude']].copy()

    # Initialize geopy with GoogleV3 geocoder
    geolocator = GoogleV3(api_key, timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1)
    geocoded_coordinates[['Latitude', 'Longitude']] = df_shared_coordinates['GEOCODE_QUERY'].progress_apply(geocode_combined_name_address)
    df.update(geocoded_coordinates[['Latitude', 'Longitude']])

    return df


def export_cleaned_data(df, output_path='cleaned_NYC_Restaurant_Database.csv'):
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Failed to save data: {e}")


def write_dicts_to_csv(data, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["center_lat", "center_lng", "radius", "points_covered"])

        # Write each dictionary as a row in the CSV
        for entry in data:
            center_lat, center_lng = entry['center']
            radius = entry['radius']
            points_covered = ', '.join(map(str, entry['points_covered']))  # Convert set to comma-separated string
            writer.writerow([center_lat, center_lng, radius, points_covered])

    return True


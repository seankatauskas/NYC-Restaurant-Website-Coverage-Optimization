import data_cleaning as dc

api_key = "AIzaSyDLpEuCMmRzOIIMwwhxO_LBmO-GqJEUy8A"

# Load Data (https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
file_path = 'DOHMH_New_York_City_Restaurant_Inspection_Results_20241021.csv'  # Replace with your actual file path
df = dc.load_restaurant_data(file_path)

# Clean and Prepare Data
df = dc.populate_full_address(df)
df = dc.populate_missing_coordinates(df, api_key)
df = dc.refine_geocode_for_restaurants(df, api_key)

# # Save the cleaned data
# dc.export_cleaned_data(df, 'cleaned_NYC_Restaurant_Database.csv')
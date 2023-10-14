
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

! pip install seaborn
import matplotlib.cm as cm
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_March23.csv")

#Understanding the data
df.head()
df.columns
df.isna().sum()
def simple_sanity_check(df):
    print('-' * 70)
    print(f'No. of Rows: {df.shape[0]}        No. of Columns: {df.shape[1]}')
    print('-' * 70)
    
    data_profile = pd.DataFrame({
        'DataType': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().mean() * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    print(data_profile)
    print('-' * 70)

simple_sanity_check(df)


########################################################### Data Cleansing_1 ########################################################### 

# This dataset contains a wealth of information, but some fields are either redundant or not directly pertinent to our analysis. To streamline the dataset and improve its focus and efficiency for our analysis, we intend to remove the following fields:

# 'Id' and 'Source': These fields do not significantly contribute to our analysis.
# 'End_Lat' and 'End_Lng': Since we already have the starting coordinates, these fields are redundant.
# 'Airport_Code': Given that all the data pertains to the USA, specifying the nearest airport code is unnecessary.
# 'Country': As previously mentioned, all the data is related to the USA, so this field does not provide additional value.
# 'Weather_Timestamp': We have other weather-related fields that are more relevant.
# 'Civil_Twilight', 'Nautical_Twilight', and 'Astronomical_Twilight': These fields may not be directly pertinent to our analysis.
# 'Timezone': This information can be derived from other relevant fields.
# By removing these fields, we aim to simplify the dataset and enhance its suitability for our analysis

# create a dataframe of Street and their corresponding accident cases
street_df = pd.DataFrame(df['Street'].value_counts()).reset_index().rename(columns={'index':'Street No.', 'Street':'Cases'})
top_ten_streets_df = pd.DataFrame(street_df.head(10))
top_ten_streets_df

# **Top 10 Accident Prone Streets in US (2016 - March 2023)**
plt.figure(figsize=(10, 6))

ax = sns.barplot(x='Cases', y='count', data=top_ten_streets_df, palette='Set2')

plt.title('Top 10 Accident Prone Streets in US (2016 - March 2023)', fontsize=16)
plt.xlabel('Accident Cases', fontsize=12)
plt.ylabel('Street', fontsize=12)
plt.show()
plt.savefig('top_10_streets.png')

### Step 1
# Columns_drop
columns_to_drop = ['ID', 'Country', 'Source', 'End_Lat', 'End_Lng', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight','Street']
df = df.drop(columns=columns_to_drop, axis=1)


### Step 2
# Fill null values of city with mode (within state)
# Calculate the mode (most frequent city) for each 'State' group
mode_cities = df.groupby('State')['City'].transform(lambda x: x.mode()[0])
# Fill missing 'City' values with the mode for their respective 'State' group
df['City'].fillna(mode_cities, inplace=True)


### step 3
# Mean fill na
# Replace missing values with the mean of their respective states
columns_to_fill = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)', 'Precipitation(in)']
for column in columns_to_fill:
    df[column] = df[column].fillna(df.groupby('State')[column].transform('mean'))
# We have decided to remove the columns 'Visibility(mi)', 'Wind_Direction', 'Weather_Condition', 'Timezone', and 'Sunrise_Sunset' from the dataset. This decision was made because these columns collectively account for approximately 3% of the entire dataset's null values. Despite the removal of rows containing null values in these columns, we are left with a substantial 7,475,297 observations in our dataset, given its large size.

# ## Step 4
# Checking for duplicates
print("Num of duplicates", df.duplicated().sum())
df = df.drop_duplicates()

### Step 5
# Handling Near Duplicates
print("No. of Weather Conditions:", len(df["Weather_Condition"].unique()))
# To view the complete list of 142 weather descriptions, run the following code
print("\nList of unique weather conditions:", list(df["Weather_Condition"].unique()))
df.loc[df["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
df.loc[df["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
df.loc[df["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
df.loc[df["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
df.loc[df["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
df.loc[df["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
df.loc[df["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
df.loc[df["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
df.loc[df["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
df.loc[df["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
df.loc[df["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan


### Step 6
# Drop null values
columns_to_check = ['Description','Visibility(mi)', 'Wind_Direction','Weather_Condition','Timezone','Sunrise_Sunset']
df.dropna(subset=columns_to_check, inplace=True)

###Step 7
#One Hot Encoding
df = df.replace([True, False], [1, 0])
simple_sanity_check(df)
print(df.info())

random_sample = df.sample(n=10000)

random_sample.columns


########################################################### Data Visualisation_1 ########################################################### 
# ## EDA 2, 3, 4
# 
# **EDA(2) -> Severity vs. Wind Speed**
# 
# **EDA(3) -> Severity vs. Precipitation**
# 
# **EDA(4) -> Severity vs. Visibility**

# Change Severity to Category
random_sample['Severity'] = random_sample['Severity'].astype('category')

# Create a set of axes and a figure
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))

# Violin plot for Wind Speed
sns.violinplot(data=random_sample, x='Severity', y='Wind_Speed(mph)', ax=axes[0], palette="husl")
axes[0].set_title('Severity vs. Wind Speed', fontsize=16)
axes[0].set_ylabel('Wind Speed (mph)', fontsize=12)
axes[0].set_xlabel('Severity', fontsize=12)

# Violin plot for Precipitation
sns.violinplot(data=random_sample, x='Severity', y='Precipitation(in)', ax=axes[1], palette="husl")
axes[1].set_title('Severity vs. Precipitation', fontsize=16)
axes[1].set_ylabel('Precipitation (in)', fontsize=12)
axes[1].set_xlabel('Severity', fontsize=12)

# Violin plot for Visibility
sns.violinplot(data=random_sample, x='Severity', y='Visibility(mi)', ax=axes[2], palette="husl")
axes[2].set_title('Severity vs. Visibility', fontsize=16)
axes[2].set_ylabel('Visibility (mi)', fontsize=12)
axes[2].set_xlabel('Severity', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()

# Change Severity back to int if needed
random_sample['Severity'] = random_sample['Severity'].astype(int)

# Count the number of cases for each state
state_counts = random_sample["State"].value_counts().reset_index()

# Rename the columns to 'state_code' and 'cases'
state_counts.rename(columns={'index': 'state_code', 'State': 'cases'}, inplace=True)

# Sort the DataFrame by case counts in descending order
state_counts = state_counts.sort_values('cases', ascending=False)


# ## EDA 5
# **Top 10 States with Highest Number of Accidents**
state_counts = df["State"].value_counts()
states = pd.DataFrame(state_counts).reset_index().sort_values('count', ascending=False)
states.rename(columns={'State':'state_code', 'count':'cases'}, inplace=True)
fig, ax = plt.subplots(figsize = (12,5))
top_10 = states[:10]
sns.barplot(x=top_10['state_code'], y=top_10['cases'], palette='Set1')
plt.title("Top 10 states with the highest number of accidents\n", fontdict = {'fontsize':16, 'color':'MidnightBlue'})
plt.ylabel("\nNumber of Accidents", fontdict = {'fontsize':12, 'color':'black'})
plt.savefig("Top 10 states with the highest number of accidents.png")
plt.show()

# ## EDA - 6
# **10 States with Lowest Number of Accidents**
## Observing which states have most accidents
fig,axs = plt.subplots(figsize = (15,8))

x = state_counts[-10:-1].index.to_list()
y = state_counts[-10:-1].values.flatten()

sns.barplot(x=x, y = y, palette='Set1')
axs.tick_params(axis = 'x')
axs.set_ylabel("Number of Accidents")
axs.set_xlabel("States")
plt.title("10 States with Lowest Number of Accidents")
plt.savefig("10 States with Lowest Number of Accidents.png")
plt.show()


# ## EDA 7
# **Top 10 Cities with Highest Number of Accidents**
city_acc_counts = pd.DataFrame(random_sample['City'].value_counts()).reset_index()
city_acc_counts.columns = ['City', "Number of Accidents"]
city_acc_counts.sort_values(by='Number of Accidents', ascending=False, inplace=True)
x = city_acc_counts['City'][:10].to_list()
y = city_acc_counts["Number of Accidents"][:10]

# Observing Top 20 cities with the most accidents
fig, axs = plt.subplots(figsize=(10, 6))
sns.barplot(x=x, y=y, ax=axs, palette='twilight')
axs.tick_params(axis='x', rotation=45)
axs.set_ylabel("Number of Accidents")
axs.set_xlabel("Cities")
plt.title("Top 10 Cities with Highest Number of Accidents")
plt.savefig("Top 10 Cities with Highest Number of Accidents.png")
plt.show()

# ## EDA - 8
# **Accidents by Various Road Features and Severity**
# Create a new dataset with boolean columns and severity
bool_severity_dataset = random_sample[['Severity', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                                      'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                                      'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']].copy()

# Calculate the number of accidents for each combination of severity and boolean columns
accidents_by_bool_severity = bool_severity_dataset.groupby(['Severity']).sum()

# Create a heatmap to visualize the relationship between accident severity and boolean columns
plt.figure(figsize=(10, 6))
sns.heatmap(data=accidents_by_bool_severity, cmap='viridis', annot=True, fmt='.1f', linewidths=.5, cbar=False)
plt.title('Accidents by Various Road Features and Severity', fontsize=16)
plt.xlabel('Road Features', fontsize=12)
plt.ylabel('Severity', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Accidents by Various Road Features and Severity.png")
plt.show()


# ## EDA - 9
# **Heatmap for Accidents on USA Map**
get_ipython().system(' pip install folium')
import folium
from folium.plugins import HeatMap
# Create a folium map centered at a specific location
m = folium.Map(location=[random_sample['Start_Lat'].mean(), random_sample['Start_Lng'].mean()], zoom_start=4.4)
# Convert your data to a list of coordinates
heatmap_data = random_sample[['Start_Lat', 'Start_Lng']].values.tolist()
# Create a HeatMap layer and add it to the folium map
HeatMap(heatmap_data).add_to(m)
# Display the folium map
m
# **Just observe how well they will overlap or match with US Accidents**
# ![This is US population Density Heatmap](https://wellsr.com/python/assets/images/2022-08-19-us-population-heatmap.PNG)
object_columns_df = df.select_dtypes(include=['object']).copy()
# column names to drop from 'df'
columns_to_drop = object_columns_df.columns
# Drop the selected columns from 'df'
df.drop(columns=columns_to_drop, inplace=True)
object_columns_df["Start_Time"].isna().sum()
# Let's start with some data manipulation, 
# Changing the datetime values to different columns, since they are in string format at this point in time.


########################################################### Data Cleansing_2 ########################################################### 


# ## Step 8
# Label Encoding
from sklearn.preprocessing import LabelEncoder

state_encoder = LabelEncoder()
county_encoder = LabelEncoder()
city_encoder = LabelEncoder()
wind_direction_encoder = LabelEncoder()
weather_condition_encoder = LabelEncoder()
sunrise_sunset_encoder = LabelEncoder()
# Encode each column and add encoded columns to the DataFrame
object_columns_df["State_LabelEncoded"] = state_encoder.fit_transform(object_columns_df["State"])
object_columns_df["County_LabelEncoded"] = county_encoder.fit_transform(object_columns_df["County"])
object_columns_df["City_LabelEncoded"] = city_encoder.fit_transform(object_columns_df["City"])
object_columns_df["Wind_Direction_LabelEncoded"] = wind_direction_encoder.fit_transform(object_columns_df["Wind_Direction"])
object_columns_df["Weather_Condition_LabelEncoded"] = weather_condition_encoder.fit_transform(object_columns_df["Weather_Condition"])
object_columns_df["Sunrise_Sunset_LabelEncoded"] = sunrise_sunset_encoder.fit_transform(object_columns_df["Sunrise_Sunset"])
encoded_state_value = 5  # Replace with the label you want to inverse transform
original_state_category = wind_direction_encoder.inverse_transform([encoded_state_value])[0]
print(f"Encoded value {encoded_state_value} corresponds to State: {original_state_category}")
object_columns_df
#  I think we made a mistake rather than headlessly converting the timezone we should have converted them to UTC
object_columns_df.Timezone.unique()
object_columns_df['Start_Time'].isna().sum()

########################################################### Data Visualisation_2 ########################################################### 

# ## EDA - 10
# **Accidents by Hour of the Day**
object_columns_df['Start_Time'] = pd.to_datetime(object_columns_df['Start_Time'] , errors='coerce')


hour_df = pd.DataFrame(object_columns_df.Start_Time.dt.hour.value_counts()).reset_index()
plt.figure(figsize=(10, 6))
plt.bar(hour_df['Start_Time'], hour_df['count'])
plt.xlabel('Hour of the Day')
plt.ylabel('Count of Accidents')
plt.title('Accidents by Hour of the Day')
plt.xticks(hour_df['Start_Time'])
plt.savefig("Accidents by Hour of the Day.png")
plt.show()

########################################################### Data Cleansing_3 ########################################################### 
### Step 9
# Feature Transformation
# changing timezones to UTC for consistency
# Convert "End_Time" to datetime objects with error handling
object_columns_df["End_Time"] = pd.to_datetime(object_columns_df["End_Time"], errors='coerce')

# Define a dictionary mapping timezones to UTC offsets (hours)
timezone_offsets = {
    'US/Eastern': -5,   # Eastern Time (ET) UTC offset
    'US/Pacific': -8,   # Pacific Time (PT) UTC offset
    'US/Central': -6,   # Central Time (CT) UTC offset
    'US/Mountain': -7,  # Mountain Time (MT) UTC offset
    # Add more timezone offsets as needed
}

# Use NumPy broadcasting to add the corresponding UTC offset to "End_Time"
object_columns_df["End_Time_UTC"] = object_columns_df["End_Time"] + pd.to_timedelta(object_columns_df["Timezone"].map(timezone_offsets), unit='h')
# Convert "Start_Time" to datetime objects
object_columns_df["Start_Time"] = pd.to_datetime(object_columns_df["Start_Time"], errors='coerce')
# Use NumPy broadcasting to add the corresponding UTC offset to "Start_Time"
object_columns_df["Start_Time_UTC"] = object_columns_df["Start_Time"] + pd.to_timedelta(object_columns_df["Timezone"].map(timezone_offsets), unit='h')
object_columns_df.columns
object_columns_df.dropna(subset=["Timezone"], inplace=True)

### Step 10
# Feature engineering
def extract_datetime_components(df, column_name):
    
    # Extract datetime components
    df[f'{column_name}_year'] = df[column_name].dt.year.astype(float)
    df[f'{column_name}_month'] = df[column_name].dt.month.astype(float)
    df[f'{column_name}_day'] = df[column_name].dt.day.astype(float)
    df[f'{column_name}_hour'] = df[column_name].dt.hour.astype(float)
    df[f'{column_name}_minute'] = df[column_name].dt.minute.astype(float)
    df[f'{column_name}_second'] = df[column_name].dt.second.astype(float)
    
    return df

object_columns_df = extract_datetime_components(object_columns_df, 'End_Time_UTC')
object_columns_df = extract_datetime_components(object_columns_df, 'Start_Time_UTC')
object_columns_df.columns
year_df = pd.DataFrame(object_columns_df.Start_Time_UTC.dt.year.value_counts()).reset_index().rename(columns={'index':'Year', 'Start_Time_UTC':'Cases'}).sort_values(by='Cases', ascending=True)
year_df['accident/day'] = year_df['count'] / 365
# Calculate cases per hour (assuming 365 days, 24 hours in a day)
year_df['accident/hour'] = year_df['count'] / (365 * 24)

########################################################### Data Visualisation_3 ########################################################### 

### EDA - 11, 12
# 
# **EDA(11) -> Average Cases of Accident/hour in US (2016- March 2023)**
# **EDA(12) -> Average Cases of Accident/Day in US (2016- March 2023)**
fig, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=year_df['Cases'], y=year_df['accident/hour'], palette='husl')
plt.title("Average Accidents per Hour in US (2016 - March 2023)")
plt.xlabel("Years")
plt.ylabel("Accidents per Hour")
plt.savefig("Average Accidents per Hour in US (2016 - March 2023).png")
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=year_df['Cases'], y=year_df['accident/day'], palette='husl')
plt.title("Average Accidents per Day in US (2016 - March 2023)")
plt.xlabel("Years")
plt.ylabel("Accidents per Day")
plt.savefig("Average Accidents per Day in US (2016 - March 2023).png")
plt.show()

import calendar
month_df = pd.DataFrame(object_columns_df.Start_Time_UTC.dt.month.value_counts()).reset_index().rename(columns={'index':'Month', 'Start_Time_UTC':'Cases'})

month_names = list(calendar.month_name)[1:]
month_df['Month'] = month_names

### EDA - 13
# **Road Accident Percentage for different months in US (2016 - March 2023)**
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors

fig, ax = plt.subplots(figsize=(10, 8))

# Change the colormap to 'viridis'
cmap = cm.get_cmap('viridis', 12)
clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(x=month_df['count'], y=month_df['Month'], palette='rainbow')
plt.title('Road Accident Percentage\nfor different months in US (2016 - March 2023)', size=20, color='grey')
plt.xlabel('Accident Cases')
plt.ylabel('Months')
plt.xlim(0, 800000)
plt.savefig("Road Accident Percentage for different months in US (2016 - March 2023).png")
plt.show()

object_columns_df.drop(
    ['State', 'County','Start_Time', 'End_Time', 'Weather_Timestamp', 'Sunrise_Sunset', 
     'Weather_Condition', 'Wind_Direction', 'Timezone', 'Airport_Code', 'Description',
    'End_Time_UTC', 'Start_Time_UTC', 'City', 'Zipcode'], axis=1, inplace=True)

# check when you run next time if 
# timestamps_UTC are still there and remove them

df = pd.concat([object_columns_df, df], axis=1)
df.head()

### EDA - 14
# **Accidents by Severity per Year**

# Perform groupby operation
severity_grouped = df.groupby(['End_Time_UTC_year', 'Severity'])['Start_Lat'].count()

# Convert the result to a DataFrame
severity_df = severity_grouped.reset_index()

# Pivot the DataFrame
severity_pivot = severity_df.pivot(index='End_Time_UTC_year', columns='Severity', values='Start_Lat')

# Plot the stacked bar chart with a different colormap
ax = severity_pivot.plot.bar(stacked=True, colormap='Set1')

# Set the title and axis labels
plt.title("Accidents by Severity per Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.savefig("Accidents by Severity per Year.png")
# Show the plot
plt.show()
data = df  

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(40, 40))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")
plt.show()

# Create a mask for upper triangular matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Fill the upper triangular matrix with True for highly correlated columns
highly_correlated = correlation_matrix.mask(mask).abs() > 0.80

# Find the columns to drop
columns_to_drop = [col for col in highly_correlated.columns if any(highly_correlated[col])]

# Print the columns that are being dropped
print("Columns to drop due to high correlation:", columns_to_drop)

# Drop the highly correlated columns from the DataFrame
data = data.drop(columns=columns_to_drop)

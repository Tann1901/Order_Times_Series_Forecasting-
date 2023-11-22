# Order_Times_Series_Forecasting

# USING PYTHON

## OVERVIEW
The objective of this project is to utilize data visualization techniques to showcase insightful findings or make specific points using the Product Demand dataset. The dataset contains historical product demand for a manufacturing company that offers a wide range of products across various categories. The company operates multiple central warehouses to distribute its products within the region it serves. As the products are manufactured in different global locations, shipping usually takes over a month via ocean transport to reach the central warehouses.
  
  a. Plot the demand for different product categories to gain insights into their demand patterns.
  
  b. Use data visualization techniques to demonstrate if there have been any changes in the demand for product categories over the years.
  
  c. Analyze and track the demand trends for the top five products in the year 2014. Present the findings visually to showcase any significant changes or patterns in their demand over the years.
  
  d. Apply a simple exponential forecasting method to predict the demand for the item with the highest demand from the category that had the highest demand in 2019.
  
## PLATFORM
Google Colab

## DATASET
ProductDemand.csv

# DATA PREPARATION
```
# Check duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicates: {duplicate_count}")
```
```
#Drop duplicates
df = df.drop_duplicates()
```
```
#Check null
df.isnull().sum()
```
```
#Change to Datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
```
```
# Drop null in Date column
df.dropna(subset=['Date'], inplace=True)
```
```
#Group Order Demand by Date, Product Code, Warehouse, Product Category
df = df.groupby(['Date', 'Product_Code', 'Warehouse','Product_Category'])['Order_Demand'].sum().reset_index()
```
```
#Split info 4 different df for 4 warehouse
warehouse_data = {}

# Dictionary to map warehouse names to custom DataFrame names
warehouse_names = {
    "St john's": "StJohn",
    "Surrey": "Surrey",
    "Oshawa": "Oshawa",
    "Brampton": "Brampton"
}

# Create separate DataFrames for each warehouse
for warehouse, data in df.groupby('Warehouse'):
    if warehouse in warehouse_names:
        df_name = warehouse_names[warehouse]
        globals()[df_name] = data.copy()
```
## DATA ANALYZATION
### Plot Category demand
```
# Create a line graph
positive_demand.plot(kind='line', marker='o', linestyle='-', color='b', figsize=(10, 6))

# Set labels and title
plt.xlabel('Product Category')
plt.ylabel('Demand')
plt.title('Positive Demand by Product Category')

# Show all x-axis labels without rotation
plt.xticks(range(len(positive_demand.index)), positive_demand.index, rotation=90)

# Show the plot
plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/65c8ba16-741a-4542-92a2-c76258c8d19c)
```
# Create a line graph
non_positive_demand.plot(kind='line', marker='o', linestyle='-', color='b', figsize=(10, 6))

# Set labels and title
plt.xlabel('Product Category')
plt.ylabel('Demand')
plt.title('0 Or Negative Demand by Product Category')

# Show all x-axis labels without rotation
plt.xticks(range(len(non_positive_demand.index)), non_positive_demand.index, rotation=90)

# Show the plot
plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/a92167da-18fe-4360-89da-97f3e4b0784d)

### Visualize Product Demand by Category changed over time
```
# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract the year from the 'Date' column
df['Year'] = df['Date'].dt.year

# Group the data by Product_Category and Year, and calculate the total demand
category_demand_by_year = df.groupby(['Product_Category', 'Year'])['Order_Demand'].sum().reset_index()

# Plot the demand for each category over the years
categories = category_demand_by_year['Product_Category'].unique()

plt.figure(figsize=(12, 6))

for category in categories:
    category_data = category_demand_by_year[category_demand_by_year['Product_Category'] == category]
    plt.plot(category_data['Year'], category_data['Order_Demand'], label=category)

# Format y-axis labels as millions
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

plt.xlabel('Year')
plt.ylabel('Demand (Millions)')
plt.title("Category Demand Over the Years")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/d92a29a9-81a5-4665-a5b4-4a6c8c73b69f)

### Top 5 products in 2014 and track their demand over the years
```
# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract the year from the 'Date' column
df['Year'] = df['Date'].dt.year

# Filter the data for the year 2014
df_2014 = df[df['Year'] == 2014]

# Group the data by Product_Code and calculate the total demand
product_demand = df_2014.groupby('Product_Code')['Order_Demand'].sum().reset_index()

# Sort the products by demand in descending order and select the top 5
top_5_products = product_demand.sort_values(by='Order_Demand', ascending=False).head(5)

# Extract the product codes of the top 5 products
top_5_product_codes = top_5_products['Product_Code'].tolist()

# Filter the original data for the top 5 products
top_5_products_data = df[df['Product_Code'].isin(top_5_product_codes)]

# Group the data by Product_Code and Year, and calculate the total demand
product_demand_by_year = top_5_products_data.groupby(['Product_Code', 'Year'])['Order_Demand'].sum().reset_index()

# Plot the demand for each product over the years
plt.figure(figsize=(12, 6))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

for product_code in top_5_product_codes:
    product_data = product_demand_by_year[product_demand_by_year['Product_Code'] == product_code]
    plt.plot(product_data['Year'], product_data['Order_Demand'], label=product_code)

plt.xlabel('Year')
plt.ylabel('Demand')
plt.title("Top 5 Products' Demand Over the Years (2014 and onwards)")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/6c55c32f-f16c-44cb-bfe5-65d6ec241ee3)

### Define the item with highest Order Demand in 2019
```
# The demand of the item with the highest demand from the category with the highest demand in 2019.
# Filter the data for the year 2019
df_2019 = df[df['Date'].dt.year == 2019]

# Group the data by 'Product_Category' and calculate the total demand in 2019
category_demand_2019 = df_2019.groupby('Product_Category')['Order_Demand'].sum()

# Identify the category with the highest demand in 2019
highest_demand_category = category_demand_2019.idxmax()

# Filter the data for the identified category
df_category = df[df['Product_Category'] == highest_demand_category]

# Group the data by 'Product_Code' and calculate the total demand in the identified category
product_demand = df_category.groupby('Product_Code')['Order_Demand'].sum()

# Identify the product with the highest demand in the identified category
highest_demand_product = product_demand.idxmax()

# Retrieve the demand of the identified product
demand_of_highest_demand_product = round(product_demand[highest_demand_product],2)

print(f"The demand of the item '{highest_demand_product}' in the category '{highest_demand_category}' in 2019 is: {demand_of_highest_demand_product}")
```
The demand of the item 'Product_1359' in the category 'Category_019' in 2019 is: 376618410.65

## TIME SERIES FORECASTING
### Visualize the different components of the time series
(a) Zoom-in to 3 years of data.
```
# shorter and longer time series
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
filtered_ts_3yrs.plot(ax=axes[0])
filtered_ts.plot(ax=axes[1])

for ax in axes:
    ax.set_xlabel('Date')
    ax.set_ylabel('Order Demand')
    ax.set_ylim(0, 2500000.0)

filtered_lm.predict(filtered_df).plot(ax=axes[1])

plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/387b1fc0-60c4-4fac-81ac-6562b708765f)

(b) Original series with overlaid quadratic trendline
```
# plot the time series
ax = filtered_ts.plot()
ax.set_xlabel('Date')
ax.set_ylabel('Order Demand')
ax.set_ylim(0, 2500000.0)
filtered_lm.predict(filtered_df).plot(ax=ax)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/aa967168-52cb-4eb1-b26c-4bc944b22823)

### Naive and seasonal naive forecasts in a 3-year validation
```
# plot forecasts and actual in the training and validation sets
ax = train_ts.plot(color='C0', linewidth=0.75, figsize=(12, 7))
valid_ts.plot(ax=ax, color='C0', linestyle='dashed', linewidth=0.75)
ax.set_xlim('2017', '2020-05')
ax.set_ylim(0, 2500000)
ax.set_xlabel('Date')

# Apply the millions formatter to the y-axis
ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
ax.set_ylabel('Order Demand (M)')

naive_pred.plot(ax=ax, color='green')
seasonal_pred.plot(ax=ax, color='orange')

# determine coordinates for drawing the arrows and lines
one_month = pd.Timedelta('31 days')
xtrain = (min(train_ts.index), max(train_ts.index) - one_month)
xvalid = (min(valid_ts.index) + one_month, max(valid_ts.index) - one_month)
xfuture = (max(valid_ts.index) + one_month, '2020-04')
xtv = xtrain[1] + 0.5 * (xvalid[0] - xtrain[1])
xvf = xvalid[1] + 0.5 * (xfuture[0] - xvalid[1])

ax.add_line(plt.Line2D(xtrain, (2350000, 2350000), color='black', linewidth=0.5))
ax.add_line(plt.Line2D(xvalid, (2350000, 2350000), color='black', linewidth=0.5))
ax.add_line(plt.Line2D(xfuture, (2350000, 2350000), color='black', linewidth=0.5))

# Add "(M)" to the labels
ax.text('2017-02', 2400000, 'Training')
ax.text('2019-03', 2400000, 'Validation')
ax.text('2020-02', 2400000, 'Future')

ax.axvline(x=xtv, ymin=0, ymax=1, color='black', linewidth=0.5)
ax.axvline(x=xvf, ymin=0, ymax=1, color='black', linewidth=0.5)

plt.show()
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/349f3799-be80-4dc9-bfc5-dc6355665fab)

### Accuracy Report
#### Validation Test
```
# Valid vs. Naive
regressionSummary(valid_ts, naive_pred)
```
Regression statistics

                      Mean Error (ME) : -499824.5614
       Root Mean Squared Error (RMSE) : 570889.7181
            Mean Absolute Error (MAE) : 543824.5614
          Mean Percentage Error (MPE) : -478.2103
Mean Absolute Percentage Error (MAPE) : 481.4025

```
 # Valid vs. Sesonal
 regressionSummary(valid_ts, seasonal_pred)
```
Regression statistics

                      Mean Error (ME) : 12425.4386
       Root Mean Squared Error (RMSE) : 320110.4730
            Mean Absolute Error (MAE) : 221714.9123
          Mean Percentage Error (MPE) : -69.5147
Mean Absolute Percentage Error (MAPE) : 122.1689

In summary, when comparing the two models:

The Seasonal model has a lower RMSE and MAPE, indicating better predictive accuracy compared to the Naive model.
Both models exhibit biases in their predictions (negative MPE for Naive and negative MPE for Seasonal), but the Seasonal model has a smaller magnitude of bias.

#### Training Set
```
# Training vs. Naive (shifted by 1 month)
regressionSummary(train_ts[1:], train_ts[:-1])
```
Regression statistics

               Mean Error (ME) : 133.9376
Root Mean Squared Error (RMSE) : 364136.2789
     Mean Absolute Error (MAE) : 259163.1420

```
# Training Vs. Naive Sesonal (shifted by 12 months)
regressionSummary(train_ts[12:], train_ts[:-12])
```
Regression statistics

               Mean Error (ME) : -642.5662
Root Mean Squared Error (RMSE) : 379147.5579
     Mean Absolute Error (MAE) : 271284.1141
```
The Naive model with a 1-month shift tends to overpredict demand on average, as indicated by the positive ME.
The Naive model with a 12-month (1-year) shift tends to underpredict demand on average, as indicated by the negative ME.
Both models exhibit significant RMSE and MAE, indicating errors in their predictions.

## Using a simple exponential forecasting method
```
fig, ax = plt.subplots(figsize=(9,4))
filtered_lm_trendseason.resid.plot(ax=ax, color='black', linewidth=0.5)
residuals_pred.plot(ax=ax, color='black', linewidth=0.5)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
ax.set_ylabel('Order Demand')
ax.set_xlabel('Date')
ax.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.5)


# Run Exponential Smoothing
# with smoothing level alpha = 0.2
expSmooth = ExponentialSmoothing(monthly_data, freq='M')
expSmoothFit = expSmooth.fit(smoothing_level=0.2)
expSmoothFit.fittedvalues.plot(ax=ax)
expSmoothFit.forecast(len(valid_ts)).plot(ax=ax, style='--', linewidth=2, color='C0')
singleGraphLayout(ax, [-1000000, 2500000], train_df, valid_df)
```
![image](https://github.com/Tann1901/Order_Times_Series_Forecasting-/assets/108020327/24c49669-ab70-4f6e-92b3-76ae2b724b7c)

Obtain the fitted values or predicted values from an exponential smoothing model.
```
expSmoothFit.fittedvalues
```
```
Date
2015-01-31     8172.854943
2015-02-28    16265.480315
2015-03-31    17023.705826
2015-04-30    15596.383393
2015-05-31    11908.672302
2015-06-30    10219.722697
2015-07-31      671.319760
2015-08-31    -4542.875037
2015-09-30    -9129.110056
2015-10-31   -13440.453049
2015-11-30   -17070.929059
2015-12-31   -19956.849417
2016-01-31    -6299.219784
2016-02-29    -3079.480867
2016-03-31     8847.576870
2016-04-30     7251.384911
2016-05-31      -57.402350
2016-06-30    -6415.158944
2016-07-31     3494.602566
2016-08-31    -6653.250002
2016-09-30     7040.245928
2016-10-31     3563.225133
2016-11-30     4143.016626
2016-12-31    10914.538676
2017-01-31    19147.476172
2017-02-28    -3232.068445
2017-03-31      884.339836
2017-04-30    10009.762602
2017-05-31     8965.857105
2017-06-30     6024.634786
2017-07-31    -1681.104295
2017-08-31     5636.146213
2017-09-30    -5199.864248
2017-10-31    -7620.306248
2017-11-30     3997.273765
2017-12-31     9878.625397
2018-01-31    -7991.475018
2018-02-28     5077.838331
2018-03-31   -14941.329256
2018-04-30   -21998.197160
2018-05-31   -11742.410160
2018-06-30    -1572.764540
2018-07-31     4563.465577
2018-08-31    10967.699764
2018-09-30    11692.142989
2018-10-31    22494.427688
2018-11-30    13092.281839
2018-12-31     3447.400791
2019-01-31     -372.856121
Freq: M, dtype: float64
```
```
# Exponential Smoothing time series of forecasted values for the future periods
expSmoothFit.forecast(len(valid_ts))
```
```
2019-02-28   -9781.541658
2019-03-31   -9781.541658
2019-04-30   -9781.541658
2019-05-31   -9781.541658
2019-06-30   -9781.541658
                 ...     
2037-09-30   -9781.541658
2037-10-31   -9781.541658
2037-11-30   -9781.541658
2037-12-31   -9781.541658
2038-01-31   -9781.541658
Freq: M, Length: 228, dtype: float64
```
```
# Residual Errors
residuals_ts
```
```
Date
2015-01-05    292732.848586
2015-01-06    200523.175878
2015-01-10    266313.915329
2015-01-11   -253894.933060
2015-01-12   -184103.369289
                  ...      
2019-01-20     40733.709224
2019-01-21   -217068.337636
2019-01-22   -240869.972336
2019-01-25   -174671.194877
2019-01-26    420527.994742
Length: 994, dtype: float64
```

# USING SAGEMAKER FOR MULTIVARIATE PREDICTIONS

Input data 

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/62b87f8a-ca63-4426-b2ed-2e8034c08e0c)

Clean data

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/ce5c3622-747d-4fa1-a44f-78364425c5bc)

Steps of Cleaning: Drop duplicate, Remove missing values 

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/262527d8-ac05-4fe0-981b-ce6501970dfe)

Configure With prediction as Order_Demand And Run Prediction

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/16e394df-890a-48df-bc38-49137f49d0cc)

Result of Prediction of Multiple Products

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/575ad258-d5de-4973-bada-b059148ab5fd)

Result of Prediction of Single Selected Product - here we select Product_1359

![image](https://github.com/Tann1901/Python-Times_Series_Demand_Forecasting/assets/108020327/051be29c-1c5f-4979-baa6-e476783106e6)


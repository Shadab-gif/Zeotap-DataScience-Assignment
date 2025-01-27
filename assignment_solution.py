import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import davies_bouldin_score

# Load datasets
customers = pd.read_csv("D:\Zeotap\Customers.csv")
products = pd.read_csv("D:\Zeotap\Products.csv")
transactions = pd.read_csv("D:\Zeotap\Transactions.csv")

# Merge datasets for analysis
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
# Region-wise sales
region_sales = merged_data.groupby('Region')['TotalValue'].sum()
region_sales.plot(kind='bar', title="Sales by Region", color='skyblue')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.savefig("region_sales.png")  # Save the plot
plt.close()

# Product category sales
category_sales = merged_data.groupby('Category')['TotalValue'].sum()
category_sales.plot(kind='bar', title="Sales by Product Category", color='green')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.savefig("category_sales.png")  # Save the plot
plt.close()

# Signup trends over time
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
signup_trend = customers.groupby(customers['SignupDate'].dt.to_period('M')).size()
signup_trend.plot(kind='line', title="Signup Trends Over Time", color='orange')
plt.xlabel('Signup Date')
plt.ylabel('Number of Signups')
plt.savefig("signup_trends.png")  # Save the plot
plt.close()
# Create customer-product matrix
customer_product_matrix = merged_data.pivot_table(
    index='CustomerID', 
    columns='ProductID', 
    values='Quantity', 
    aggfunc='sum', 
    fill_value=0
)

# Compute cosine similarity
similarity_matrix = cosine_similarity(customer_product_matrix)

# Generate lookalike recommendations
lookalike_map = {}
customer_ids = customer_product_matrix.index.tolist()

for idx, customer_id in enumerate(customer_ids[:20]):  # First 20 customers
    similar_customers = sorted(
        enumerate(similarity_matrix[idx]),
        key=lambda x: x[1],
        reverse=True
    )[1:4]  # Exclude self
    lookalike_map[customer_id] = [
        (customer_ids[i], score) for i, score in similar_customers
    ]

# Save recommendations to CSV
lookalike_df = pd.DataFrame({
    'CustomerID': lookalike_map.keys(),
    'Lookalikes': [str(v) for v in lookalike_map.values()]
})
lookalike_df.to_csv("Lookalike.csv", index=False)
# Aggregated customer metrics
customer_metrics = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',    # Total spending
    'Quantity': 'sum',      # Total quantity purchased
    'Price_y': 'mean'       # Average price
}).reset_index()

# Normalize data for clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(customer_metrics[['TotalValue', 'Quantity', 'Price_y']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_metrics['Cluster'] = kmeans.fit_predict(normalized_data)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(normalized_data, customer_metrics['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")

# Save clustering results
customer_metrics.to_csv("Clustering.csv", index=False)

# Visualize customer clusters
sns.scatterplot(
    x=customer_metrics['TotalValue'],
    y=customer_metrics['Quantity'],
    hue=customer_metrics['Cluster'],
    palette='viridis'
)
plt.title("Customer Clusters")
plt.xlabel("TotalValue")
plt.ylabel("Quantity")
plt.savefig("customer_clusters.png")  # Save the plot
plt.close()

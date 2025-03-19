import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to load the data
def load_data():
    images = np.load('/Users/sankethkaruturi/Desktop/CS434 HW4 Decision Trees and K-Means Clustering /HW4_CS434 2/img.npy')
    hog_features = np.load('/Users/sankethkaruturi/Desktop/CS434 HW4 Decision Trees and K-Means Clustering /HW4_CS434 2/hog.npy')
    return images, hog_features

# Function to run k-means clustering
def run_kmeans(hog_features, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(hog_features)
    return kmeans

# Function to visualize the clusters
def visualize_clusters(images, assignments, k):
    plt.figure(figsize=(15, 10))
    for i in range(k):
        cluster_images = images[assignments == i]
        # Show up to 50 samples from each cluster
        for j in range(min(len(cluster_images), 50)):
            plt.subplot(k, 50, i * 50 + j + 1)
            plt.imshow(cluster_images[j], cmap='gray')
            plt.axis('off')
    plt.show()

# Function to calculate SSE
def calculate_sse(hog_features, kmeans):
    distances = kmeans.transform(hog_features)
    closest_cluster = np.argmin(distances, axis=1)
    sse = np.sum(distances[range(distances.shape[0]), closest_cluster]**2)
    return sse

# Load the images and HOG features
images, hog_features = load_data()

# Run k-means with k=10 and visualize the results
kmeans = run_kmeans(hog_features, k=10)
visualize_clusters(images, kmeans.labels_, k=10)

# Calculate and print the SSE for k=10
sse_k10 = calculate_sse(hog_features, kmeans)
print(f'SSE for k=10: {sse_k10}')

# Run k-means with a different value of k and visualize the results
# You can change this value to find a suitable k
k = 100  # Example value for k
kmeans = run_kmeans(hog_features, k)
visualize_clusters(images, kmeans.labels_, k)

# Calculate and print the SSE for your chosen k
sse_k = calculate_sse(hog_features, kmeans)
print(f'SSE for k={k}: {sse_k}')

# Compare the SSE values
print('Comparing SSE values:')
print(f'SSE for k=10: {sse_k10}')
print(f'SSE for k={k}: {sse_k}')


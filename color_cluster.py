#
#  Use k-means clustering to find the most-common colors in an image
#
import cv2
import numpy as np
from sklearn.cluster import KMeans

def make_histogram(cluster):
  """
  Count the number of pixels in each cluster
  :param: KMeans cluster
  :return: numpy histogram
  """
  numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
  hist, _ = np.histogram(cluster.labels_, bins=numLabels)
  hist = hist.astype('float32')
  hist /= hist.sum()
  return hist

def cluster(img, dim, num_clusters):
  img = cv2.resize(img, (dim, dim))
  height, width, _ = np.shape(img)
  image = img.reshape((height * width, 3))
  num_clusters = 5
  clusters = KMeans(n_clusters=num_clusters, n_init="auto")
  clusters.fit(image)

  histogram = make_histogram(clusters)
  combined = zip(histogram, clusters.cluster_centers_)
  combined = sorted(combined, key=lambda x: x[0], reverse=True)
  return combined

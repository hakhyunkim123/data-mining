from math import sqrt
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from load_mnist import load_mnist_dataset

class Kmeans:
    def __init__(self, dataset, k, max_iterations=50) :
        self.k = k
        self.dataset = dataset
        self.max_iterations = max_iterations
    
    def as_numpy_array(self) :
        self.np_dataset = np.asarray(self.dataset, dtype=np.float32)
        self.np_centroids = np.asarray(self.centroids, dtype=np.float32)
    
    def init_centroid(self, seed) :
        random.seed(1234+seed)
        self.centroids = random.sample(self.dataset, self.k)

    def assign_cluster(self) :
        self.clusters = {}
        for i in range(self.k) :
            self.clusters[i] = []

        for data in self.np_dataset :
            # get euclidean distance
            distances = [np.linalg.norm(data - centroid) for centroid in self.np_centroids] 
            idx = distances.index(min(distances)) # select minimum value of distance
            self.clusters[idx].append(data) # assign data to cluster
        
    def update_centroid(self) :
        for idx in range(self.k) :
            self.np_centroids[idx] = np.mean(self.clusters[idx], axis=0)

    """
    clustering evaluation.
    loss function - SSE(Sum of Sqaured Distance)
    """
    def cal_SSE(self) :
        sse = 0
        for i in range(self.k) :
            for j in range(len(self.clusters[i])) :
                sse += np.linalg.norm(self.clusters[i][j] - self.np_centroids[i])
        return sse

    def process(self) :
        self.as_numpy_array()

        # first k-means clustering
        self.assign_cluster()
        self.update_centroid()
        self.sse_seq = []
        self.sse_seq.append(self.cal_SSE())

        # k-means iterations
        for i in range(1, self.max_iterations) :
            self.assign_cluster()
            self.update_centroid()
            tmp_sse = self.cal_SSE()
            if (min(self.sse_seq) - tmp_sse) == 0 :
                break # loss function value에 변화가 없으면 클러스터링 종료
            else :
                self.sse_seq.append(tmp_sse)

    # print k-means clustering information
    def report(self) :
        print('number of clusters: {}'.format(self.k))
        print('loss value of best solution(SSE): {}'.format(min(self.sse_seq)))
        print('sequence of loss values')
        for i in range(len(self.sse_seq)) : 
            print('k-means iteration {} : {}'.format(i+1, self.sse_seq[i]))
        for i in range(self.k) :
            print('cluster {} : {} elements'.format(i, len(self.clusters[i])))


def my_kmeans(dataset, k, init) :
    sse_list = []
    results = []
    seed_list = random.sample(range(1, 1000), init)
    for i in range(init) :
        kmeans_algorithm = Kmeans(dataset, k)
        kmeans_algorithm.init_centroid(seed=seed_list[i])
        kmeans_algorithm.process()

        # loss function values list (initialization iterations)
        sse_list.append(min(kmeans_algorithm.sse_seq))
        # kmeans result list
        results.append(kmeans_algorithm)
    
    """
    best_solution -> best k-means clustering result.
    best_solution.report() -> k-means clustering 결과 출력
    best_solution.k -> clustering 개수
    best_solution.sse_seq -> sequence of loss values

    best_solution.sse_seq's last element(min(best_solution.sse_seq)) -> loss value of the clustering
    -> MNIST Dataset에 대한 k-means 클러스터링은 (1000개의 데이터)
       iteration을 반복할때 마다 loss function value는 점점 감소하거나 변화가 없다.
    """
    best_solution_idx = sse_list.index(min(sse_list))
    best_solution = results[best_solution_idx] 
    return best_solution, sse_list

def variation_of_k_test() :
    dataset = mnist_images[:1000]
    results = []
    for i in range(20) :
        print('iter {}'.format(i+1))
        best_solution, sse_list = my_kmeans(dataset, k=i+1, init=10)
        results.append(min(sse_list))

    dif_list = []
    for i in range(len(results)-1) :
        dif_list.append(results[i] - results[i+1])

    # print result 
    # for i in range(len(results)) :
    #    print('k: {} -> {}'.format(i+1, results[i]))

    # for i in range(len(dif_list)) :
    #    print('{}~{} -> {}'.format(i+2, i+1, dif_list[i]))

    """
    plt.title("k-means algorithm simulation")
    plt.plot(results, 'b', dif_list, 'r')
    plt.xlabel('k/interval')
    plt.ylabel('sse')
    plt.show()
    """
    
def default_test() :
    dataset = mnist_images[:1000]
    best_solution, sse_list = my_kmeans(dataset, k = 10, init = 10)
    best_solution.report() # k-means result imformation

    """
    plt.title("N initialzation loss values")
    plt.plot(sse_list, 'b')
    plt.xlabel('number of initializations')
    plt.ylabel('sse')
    plt.show()

    plt.title("k-means iterations loss values")
    plt.plot(best_solution.sse_seq, 'r')
    plt.xlabel('iteration')
    plt.ylabel('sse')
    plt.show()
    """

mnist_images, mnist_labels = load_mnist_dataset()

# choose one
default_test()
# variation_of_k_test()
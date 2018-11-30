class MNIST_data :
    def __init__(self, image, label) :
        self.image = image
        self.label = label

def load_mnist_dataset() :
    from mnist import MNIST
    mn_data = MNIST('./mnist')

    images, labels = mn_data.load_training()
    print('loading mnist dataset complete.')
    return images, labels

def mnist_data(mn_images, mn_labels) :
    dataset = []
    for i in range(len(mn_images)) :
        mn_data = MNIST_data(mn_images[i], mn_labels[i])
        dataset.append(mn_data)
    return dataset
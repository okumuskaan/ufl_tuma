"""Dataset loading, preprocessing, caching and partitioning helpers.

Supports MNIST, FashionMNIST and CIFAR-10. Downloads and caches processed
data under data_dir/processed_data to avoid repeated transforms. Provides
    DataHandler for loading and distributing data to clients and
    DataPartitioner for several partitioning strategies (IID, Dirichlet, custom).
"""
import torch 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from random import Random
import numpy as np 
from numpy.random import RandomState 
import os
from os.path import exists


class DataHandler:
    def __init__(self, dataset, data_dir, alpha, random_seed=30, include_server=True, num_clients=None):
        self.dataset = dataset
        self.data_dir = data_dir
        self.alpha = alpha
        self.random_seed = random_seed
        self.num_clients = num_clients
        self.include_server = include_server

        if dataset not in ["mnist", "cifar10", "fmnist"]:
            raise ValueError("Invalid dataset")
        elif dataset in ["mnist", "fmnist"]:
            if dataset == "mnist":
                self.train_dataset, self.test_dataset = self.load_mnist_flat()
            else:
                self.train_dataset, self.test_dataset = self.load_fmnist_flat()
        elif dataset == "cifar10":
            self.train_dataset, self.test_dataset = self.load_cifar10()

    def download_dataset(self, dataset_cls):
        apply_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
        train_data_global = dataset_cls(
            root=self.data_dir + "/dataset/", train=True, transform=apply_transform, download=True
        )
        test_data_global = dataset_cls(
            root=self.data_dir + "/dataset/", train=False, transform=apply_transform, download=True
        )
        return train_data_global, test_data_global

    def load_mnist_flat(self):
        if not exists(self.data_dir + "/processed_data/mnist_flat/test_data_global.pt"):
            train_data_global, test_data_global = self.download_dataset(datasets.MNIST)
            os.makedirs(self.data_dir + "/processed_data/mnist_flat/", exist_ok=True)
            torch.save(train_data_global, self.data_dir + "/processed_data/mnist_flat/train_data_global.pt")
            torch.save(test_data_global, self.data_dir + "/processed_data/mnist_flat/test_data_global.pt")
            self.input_shape = train_data_global.data[0].shape
            self.output_dim = train_data_global.targets.unique().shape[0]
        else:
            train_data_global = torch.load(self.data_dir + "/processed_data/mnist_flat/train_data_global.pt", weights_only=False)
            test_data_global = torch.load(self.data_dir + "/processed_data/mnist_flat/test_data_global.pt", weights_only=False)
            self.input_shape = train_data_global.data[0].shape
            self.output_dim = train_data_global.targets.unique().shape[0]
        return train_data_global, test_data_global

    def load_fmnist_flat(self):
        if not exists(self.data_dir + "/processed_data/fmnist_flat/test_data_global.pt"):
            train_data_global, test_data_global = self.download_dataset(datasets.FashionMNIST)
            os.makedirs(self.data_dir + "/processed_data/fmnist_flat/", exist_ok=True)
            torch.save(train_data_global, self.data_dir + "/processed_data/fmnist_flat/train_data_global.pt")
            torch.save(test_data_global, self.data_dir + "/processed_data/fmnist_flat/test_data_global.pt")
            self.input_shape = train_data_global.data[0].shape
            self.output_dim = train_data_global.targets.unique().shape[0]
        else:
            train_data_global = torch.load(self.data_dir + "/processed_data/fmnist_flat/train_data_global.pt", weights_only=False)
            test_data_global = torch.load(self.data_dir + "/processed_data/fmnist_flat/test_data_global.pt", weights_only=False)
            self.input_shape = train_data_global.data[0].shape
            self.output_dim = train_data_global.targets.unique().shape[0]
        return train_data_global, test_data_global
        
    def load_cifar10(self):
        """
        Download (if needed), transform to [N,3,32,32] once, normalize to [-1,1],
        and cache as torch tensors while preserving .data and .targets attributes.
        """
        proc_dir = os.path.join(self.data_dir, "processed_data", "cifar10")
        os.makedirs(proc_dir, exist_ok=True)
        train_path = os.path.join(proc_dir, "train_data_global.pt")
        test_path  = os.path.join(proc_dir, "test_data_global.pt")

        if not exists(test_path):
            # --- 1. Load raw CIFAR-10 without transforms
            raw_train = datasets.CIFAR10(
                root=os.path.join(self.data_dir, "dataset"),
                train=True,
                download=True
            )
            raw_test = datasets.CIFAR10(
                root=os.path.join(self.data_dir, "dataset"),
                train=False,
                download=True
            )

            # --- 2. One-time transform to tensor [C,H,W] and normalize to [-1,1]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data   = torch.stack([transform(img) for img in raw_train.data])  # [N,3,32,32]
            train_labels = torch.tensor(raw_train.targets, dtype=torch.long)
            test_data    = torch.stack([transform(img) for img in raw_test.data])
            test_labels  = torch.tensor(raw_test.targets, dtype=torch.long)

            # --- 3. Save processed tensors
            torch.save((train_data, train_labels), train_path)
            torch.save((test_data,  test_labels),  test_path)
        else:
            # --- 4. Reload processed tensors
            train_data, train_labels = torch.load(train_path, weights_only=False)
            test_data,  test_labels  = torch.load(test_path, weights_only=False)

        # --- 5. Reconstruct CIFAR10-like objects with no transform
        train_dataset = datasets.CIFAR10(
            root=os.path.join(self.data_dir, "dataset"),
            train=True,
            transform=None
        )
        test_dataset = datasets.CIFAR10(
            root=os.path.join(self.data_dir, "dataset"),
            train=False,
            transform=None
        )

        # Overwrite with preprocessed tensors (already normalized, CHW)
        train_dataset.data    = train_data
        train_dataset.targets = train_labels
        test_dataset.data     = test_data
        test_dataset.targets  = test_labels

        # Record shapes for model construction
        self.input_shape = train_data[0].shape   # (3, 32, 32)
        self.output_dim  = len(torch.unique(train_labels))

        return train_dataset, test_dataset

    def distribute_data(self, num_clients):
        num_users = num_clients + 1 if self.include_server else num_clients

        partition_sizes = [1.0/num_users for _ in range(num_users)]
        partitioner = DataPartitioner(self.train_dataset, partition_sizes, seed=self.random_seed, isNonIID=True, alpha=self.alpha, dataset=self.dataset)
        train_partitions = partitioner.partitions

        partitioner_ = DataPartitioner(self.test_dataset, partitioner.ratio, seed=self.random_seed, 
                                            isNonIID=False, alpha=0, dataset=self.dataset)
        
        test_partitions = partitioner_.partitions

        self.client_indices_train = {}
        self.client_indices_test = {}
        for i in range(num_clients):
            self.client_indices_train[i] = train_partitions[i]
            self.client_indices_test[i] = test_partitions[i]
        
        self.server_index_train = train_partitions[num_users-1] if self.include_server else None
        self.server_index_test = test_partitions[num_users-1] if self.include_server else None

        self.ratio = partitioner.ratio
                
        return self.client_indices_train, self.client_indices_test, self.server_index_train, self.server_index_test


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chunks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], rnd=0, seed=1234, isNonIID=False, alpha=0,
                 dataset=None, print_f=50):
        self.data = data
        self.dataset = dataset

        if isNonIID:
            self.partitions, self.ratio, self.dat_stat, self.endat_size = self.__getDirichletData__(data, sizes, alpha)

        else:
            self.partitions = [] 
            self.ratio = sizes
            rng = Random() 
            rng.seed(seed) # seed is fixed so same random number is generated
            data_len = len(data) 
            indexes = [x for x in range(0, data_len)] 
            rng.shuffle(indexes)    # Same shuffling (with each seed)

            for frac in sizes: 
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed, alpha):
        # labelList = data.targets
        labelList = np.array([data[i][1] for i in range(len(data))])  # alternative for above
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]

        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum

        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))

        # majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        majorLabelNumPerPartition = 2
        basicLabelRatio = alpha
        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]

        return partitions

    def __getDirichletData__(self, data, psizes, alpha):
        n_nets = len(psizes)
        K = 10
        # labelList = np.array(data.targets)
        labelList = np.array([data[i][1] for i in range(len(data))])  # alternative for above
        min_size = 0
        N = len(labelList)
        rann = RandomState(2020)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                rann.shuffle(idx_k)
                proportions = rann.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            rann.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)

        return idx_batch, weights, net_cls_counts, np.sum(local_sizes)

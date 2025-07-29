import numpy as np
import pickle
import os
import tarfile
import urllib.request

def download_cifar10(dataset_dir='./cifar10'):
    """
    下载并解压CIFAR-10数据集
    
    参数:
    - dataset_dir: 数据集保存目录
    """
    CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 检查数据集压缩包是否存在
    download_path = os.path.join(dataset_dir, "cifar-10-python.tar.gz")
    
    # 如果当前目录下没有压缩包，检查父目录
    if not os.path.exists(download_path):
        parent_dir_path = os.path.join(os.path.dirname(dataset_dir), "cifar-10-python.tar.gz")
        if os.path.exists(parent_dir_path):
            print(f"Found CIFAR-10 archive in parent directory, using file: {parent_dir_path}")
            download_path = parent_dir_path
        else:
            print("Downloading CIFAR-10 dataset...")
            urllib.request.urlretrieve(CIFAR10_URL, download_path)
            print("Download finished!")
    
    # 解压缩数据集
    extract_dir = os.path.join(dataset_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=dataset_dir)
        print("Extraction finished!")
    
    return extract_dir

def load_batch(file_path):
    """
    加载CIFAR-10批次数据
    
    参数:
    - file_path: 批次文件路径
    
    返回:
    - 数据和标签
    """
    with open(file_path, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')
    
    # 处理数据
    X = batch_data[b'data']
    y = np.array(batch_data[b'labels'])
    
    # 转换数据形状为 [N, 3, 32, 32] 或 [N, 32, 32, 3]
    X = X.reshape(X.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    
    return X, y

def load_cifar10(dataset_dir='./cifar10', normalize=True, flatten=True, one_hot=False):
    """
    加载CIFAR-10数据集
    
    参数:
    - dataset_dir: 数据集目录
    - normalize: 是否将像素值标准化到[0,1]
    - flatten: 是否将图像展平为一维向量
    - one_hot: 是否使用one-hot编码标签
    
    返回:
    - (训练图像, 训练标签), (测试图像, 测试标签)
    """
    # 确保数据集已下载
    extract_dir = download_cifar10(dataset_dir)
    
    # 加载训练数据
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        X_batch, y_batch = load_batch(batch_file)
        X_train.append(X_batch)
        y_train.append(y_batch)
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # 加载测试数据
    test_file = os.path.join(extract_dir, 'test_batch')
    X_test, y_test = load_batch(test_file)
    
    # 标准化
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    # 展平
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    # One-hot编码
    if one_hot:
        y_train_one_hot = np.zeros((y_train.size, 10))
        y_train_one_hot[np.arange(y_train.size), y_train] = 1
        y_train = y_train_one_hot
        
        y_test_one_hot = np.zeros((y_test.size, 10))
        y_test_one_hot[np.arange(y_test.size), y_test] = 1
        y_test = y_test_one_hot
    
    return (X_train, y_train), (X_test, y_test)

class DataSplitter:
    """处理训练集、验证集划分的工具类"""
    @staticmethod
    def train_val_split(X, y, val_size=0.1, shuffle=True):
        """
        将数据集划分为训练集和验证集
        
        参数:
        - X: 特征
        - y: 标签
        - val_size: 验证集比例
        - shuffle: 是否打乱数据
        
        返回:
        - X_train, X_val, y_train, y_val
        """
        if shuffle:
            # 生成索引并打乱
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        # 计算验证集大小
        val_samples = int(X.shape[0] * val_size)
        
        # 划分数据
        X_train, X_val = X[val_samples:], X[:val_samples]
        y_train, y_val = y[val_samples:], y[:val_samples]
        
        return X_train, X_val, y_train, y_val

class MiniBatchLoader:
    """Mini-batch数据加载器"""
    def __init__(self, X, y, batch_size=100, shuffle=True):
        """
        初始化
        
        参数:
        - X: 特征数据
        - y: 标签数据
        - batch_size: 批量大小
        - shuffle: 是否在每个epoch开始时打乱数据
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)
        
        # 初始时打乱数据顺序
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_batch = 0
    
    def __iter__(self):
        """返回迭代器自身"""
        self.current_batch = 0
        
        # 在每个epoch开始时打乱顺序
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        return self
    
    def __next__(self):
        """返回下一个批次的数据"""
        if self.current_batch >= self.n_batches:
            # 迭代结束
            raise StopIteration
        
        # 计算当前批次的索引范围
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # 获取当前批次的数据
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        self.current_batch += 1
        
        return X_batch, y_batch 
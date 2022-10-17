from torch.utils.data import Dataset, DataLoader


# Dataset 是抽象类，需要继承后自己定义，指向数据集
class MyDataset(Dataset):
    def __init__(self):
        pass

    # 能根据索引取出数据集中的数据
    def __getitem__(self, item):
        pass

    # 能获取数据集的数量
    def __len__(self):
        pass

dataset = MyDataset()
# DataLoader 用于加载数据，打乱等功能
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

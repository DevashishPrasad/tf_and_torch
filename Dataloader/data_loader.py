# Make the dataset mapping based for dataset loader
class ExDataset(Dataset):
    def __init__(self):
        self.data = final_imgs
        self.target = final_labels
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x,y
    
    def __len__(self):
        return len(self.data)

Edataset = ExDataset()

# 10% Train - Test Split
train_set, test_set = torch.utils.data.random_split(Edataset, [20250, 2250])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True,num_workers=2)

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

class ImageDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform) 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

        #image = transforms.Compose([
         #   transforms.Resize((128, 128)),
          #  transforms.ToTensor(),])(Image.open(image_path))

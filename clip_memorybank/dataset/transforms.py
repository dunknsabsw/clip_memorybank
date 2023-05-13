from torchvision import transforms


class TransformsTrain:
    def __init__(self, size):
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.train_transforms(x)


class TransformsVal:
    def __init__(self, size):
        self.valid_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.valid_transforms(x)
    

class TransformsTrain2:
    def __init__(self, size):
        self.train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(size, scale=(0.5, 1.5)),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomResizedCrop(size, scale=(0.5, 1), ratio=(1 / 3, 3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.train_transforms(x)


class TransformsVal2:
    def __init__(self, size):
        self.valid_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.valid_transforms(x)
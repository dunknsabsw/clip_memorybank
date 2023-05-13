import os
from torch.utils.data import Dataset
import numpy as np
import random
import PIL


class baidu_dataset(Dataset):
    """baidu CVPR workshop challenge 2023 dataset loader for single sentence
    Attributes:
        txt_path:  image file names and captions
        image_path: image directory
        tokenizer: tokenize the word
        max_words: the max number of word
        feature_framerate: frame rate for sampling video
        max_frames: the max number of frame
        image_resolution: resolution of images
    """
    def __init__(
            self,
            data_path,
            tokenizer,
            max_words=30,
            train_transform=None,
            val_transform=None,
            shuffle=False,
            mode='train',
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.mode = mode
        self.train_transform = train_transform
        self.val_transform = train_transform if val_transform is None else val_transform
        self.transform = self.train_transform if self.mode == "train" else val_transform
        self.shuffle = shuffle
        self.data = []
        # 打开数据
        if self.mode == 'train':
            img = os.listdir(os.path.join(self.data_path, 'train_images/'))
            txt = os.path.join(self.data_path, 'train_label.txt')
        elif self.mode == 'test':
            img = os.listdir(os.path.join(self.data_path, 'test_images/'))
            txt = os.path.join(self.data_path, 'test_text.txt')
        else:
            img = os.listdir(os.path.join(self.data_path, 'val_images/'))
            txt = os.path.join(self.data_path, 'val_label.txt')

        f = open(txt, 'r')
        if self.mode == 'train':
            for line in f.readlines():
                line = line.strip()
                items = line.split('$')
                name, _, text = items
                image_path = os.path.join(self.data_path, 'train_images/', name)
                self.data.append([image_path, text, name])

        # start and end token
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def _get_text(self, caption):
        """get tokenized word feature
        Args:
            caption: caption

        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text
        """
        # tokenize word
        words = self.tokenizer.tokenize(caption)
        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        # add end token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def __getitem__(self, index):
        image, text, name = self.data[index]
        description = text
        text, _, _, = self._get_text(description)

        try:
            img = PIL.Image.open(image).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        except:
            print(f"An exception occurred trying to load file {image}.")
            print(f"Skipping index {index}")
            return self.skip_sample(index)

        return text, img, index

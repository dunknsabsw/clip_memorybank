import sys
import os
import torch
from torch.utils.data import DataLoader
from dataset.dataset_baidu import baidu_dataset
from dataset.transforms import TransformsTrain, TransformsVal


sys.path.append(os.path.dirname(__file__) + os.sep + '../')


def dataloader_baidu_train(args, tokenizer):
    dataset = baidu_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        train_transform=TransformsTrain(224),
        val_transform=TransformsVal(224),
        mode="train"
    )

    if args.gpu_type == "ddp":
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(dataset), train_sampler


def dataloader_baidu_infer(args, tokenizer):
    dataset = baidu_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        mode=args.infer_mode
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(dataset)

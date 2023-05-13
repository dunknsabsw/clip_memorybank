import numpy as np
import time
import random
from tqdm import tqdm
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling_open import CLIP4Clip
from modules.optimization import BertAdam
from utils.config import get_args
from utils.utils import get_logger
from dataset.dataloader import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def set_seed_logger(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return args


# 参数设置
args = get_args()
# 修改：数据集 & 一些路径
args.dataset_name = "cvpr23"
args.data_path = "/root/autodl-tmp/dataset/train"
args.model_path = "/root/autodl-tmp/model/ViT-B-32.pt"
args.checkpoint = "/root/autodl-tmp/checkpoint/"
args.cross_model = "cross-base"
# 修改：一些学习中的参数
args.lr = 1e-7
args.vision_lr = 1e-7
args.text_lr = 1e-7
args.max_words = 32
# 修改：日志汇报频率
args.n_display = 1
# 修改：device相关
args = set_seed_logger(args)
args.n_gpu = torch.cuda.device_count()
# 修改：基本训练参数
args.batch_size = 256
args.epochs = 20
# 修改：memory-bank参数
args.K = 65536
args.T = 1.0
args.M = 0.5
args.memory_bank = True
args.mlp = True
# 修改：读取的模型号码
args.model_num = 0
# 修改：保存模型的频率(每x个epoch保存一次)
args.save_per_epoch = 5

# 日志初始化
global logger
logger = get_logger(os.path.join(args.checkpoint, "log.txt"))


def init_model(args):
    # 模型路径
    model_file = os.path.join(args.checkpoint, "T_{}_epoch_{}".format(int(args.T), args.model_num))
    # 加载模型
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None
        if args.local_rank == 0:
            logger.info("Model loaded fail %s", model_file)
    # initialize
    model = CLIP4Clip.from_pretrained(args, state_dict=model_state_dict)
    # cuda
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    return model


def save_model(epoch, args, model):
    # 待存模型
    model_save = model.module if hasattr(model, 'module') else model
    # 目标路径
    file_save = os.path.join(args.checkpoint, "T_{}_epoch_{}_mlp".format(1, epoch + 1))
    # 检查-保存
    if not os.path.exists(args.checkpoint):
        os.makedirs(file_save)
    elif not os.path.isfile(file_save):
        torch.save(model_save.state_dict(), file_save)
    # 日志
    logger.info("Model saved to %s", file_save)
    return file_save


def init_optimizer(args, model, num_train_optimization_steps):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    decay_param = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    for n, _ in param_optimizer:
        continue

    # need to decay
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in decay_param)]

    # don't need to dacay
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in decay_param)]

    # parameters of CLIP
    decay_clip_visual_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n and "clip.visual" in n]
    decay_clip_text_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n and "clip.visual" not in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    # parameters which don't belong to CLIP
    no_decay_clip_visual_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n and "clip.visual" in n]
    no_decay_clip_text_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n and "clip.visual" not in n]
    no_decay_noclip_visual_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    # define the learning rate and weight decay for different modules
    # the learning rate of pre-trained clip modules should be defined a quite small for convergence
    # the learning rate of proposed blocks can be defined larger than learning rate of clip
    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_visual_param_tp], 'weight_decay': weight_decay, 'lr': args.vision_lr},
        {'params': [p for n, p in decay_clip_text_param_tp], 'weight_decay': weight_decay, 'lr': args.text_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay, "lr": args.lr},
        {'params': [p for n, p in no_decay_clip_visual_param_tp], 'weight_decay': 0.0, 'lr': args.vision_lr},
        {'params': [p for n, p in no_decay_clip_text_param_tp], 'weight_decay': 0.0, 'lr': args.text_lr},
        {'params': [p for n, p in no_decay_noclip_visual_param_tp], 'weight_decay': 0.0, 'lr': args.lr},
    ]

    for name, parameter in model.named_parameters():
        if "clip." in n and "clip.visual" in n and args.frozen_clip_image:
            parameter.requires_grad = False
        elif "clip." in n and "clip.visual" not in n and args.frozen_clip_text:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # initialize the optimizer
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    return optimizer, model


def train_epoch(epoch, args, model, train_dataloader, optimizer, global_step, local_rank=0):

    torch.cuda.empty_cache()
    # train model
    model.train()

    log_step = args.n_display  # 日志汇报频率
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # cuda一下
        if args.n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.cuda() for t in batch)
        # 图、文、序号
        cap, img, index = batch
        # 前传，计算损失
        loss = model(cap, img, index)
        # calculate the loss for display
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        # calculate the loss if the gradient accmulation exists
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        # backward for the model
        loss.backward()
        # counting the loss
        total_loss += float(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # issues introduced from https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            # logging the loss per batch
            global_step += 1
            # local_rank: the id of machine, where the output of machine 0 should be logged
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1, len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]), float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step


def main():
    args.rank = args.local_rank
    # load tokenizer
    tokenizer = ClipTokenizer()
    # load dataloader
    train_dataloader, train_length, train_sampler = dataloader_baidu_train(args, tokenizer)
    args.bank_size = train_length

    # init model and load model if necessary
    model = init_model(args)

    # total number of optimization
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    # load optimizer the distributed model
    optimizer, model = init_optimizer(args, model, num_train_optimization_steps)

    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

    # training
    global_step = 0
    for epoch in range(args.epochs):
        # initialize the sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # for model optimization in one epoch
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, optimizer, global_step,
                                           local_rank=args.local_rank)

        # record the end of epoch and save epoch according to args.save_per_epoch
        if args.local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            if (epoch + 1) % args.save_per_epoch == 0:
                save_model(epoch, args, model)


if __name__ == "__main__":
    main()

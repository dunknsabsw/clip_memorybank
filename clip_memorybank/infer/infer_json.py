import numpy as np
import random
from tqdm import tqdm
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling_open import CLIP4Clip
from PIL import Image, ImageOps
from utils.config import get_args
from utils.utils import get_logger
from dataset.dataloader import *
from utils.file_io import PathManager
import json
from dataset.transforms import TransformsVal
import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def set_seed_logger(args):
    global logger
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


# 参数
args = get_args()
args.checkpoint = "/root/autodl-tmp/checkpoint/"
args.model_path = "/root/autodl-tmp/model/ViT-B-32.pt"
args.infer_mode = "val"
args.data_path = "/root/autodl-tmp/dataset/test" if args.infer_mode == "test" else "/root/autodl-tmp/dataset/val"
args.T = 1.0
args.model_num = 20
args.memory_bank = False
args.mlp = True

args = set_seed_logger(args)

# 测试数据路径(模型加载路径line70//json保存路径line215)
img_path = '/root/autodl-tmp/dataset/test/test_images/' \
    if args.infer_mode == "test" else '/root/autodl-tmp/dataset/val/val_images/'
txt_path = '/root/autodl-tmp/dataset/test/test_text.txt' \
    if args.infer_mode == "test" else '/root/autodl-tmp/dataset/val/val_label.txt'

# 日志初始化
global logger
logger = get_logger(os.path.join(args.checkpoint, "log.txt"))

# 分词器
tokenizer = ClipTokenizer()


def init_device(args, local_rank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    n_gpu = 1
    args.n_gpu = n_gpu
    return device, n_gpu


def init_model(args, device):
    # 模型路径
    model_file = os.path.join(args.checkpoint, "T_{}_epoch_{}_mlp".format(int(args.T), args.model_num))
    # 加载模型
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None
        if args.local_rank == 0:
            logger.info("Model loaded fail %s", model_file)
    # 删掉memory-bank
    if args.memory_bank:
        del model_state_dict['seq_bank'], model_state_dict['vis_bank']
    # Prepare model
    model = CLIP4Clip.from_pretrained(args, state_dict=model_state_dict)
    model.to(device)
    return model


def read_img(img_name, format=None):
    with PathManager.open(img_name, "rb") as file_img:
        # 打开图片
        img = Image.open(file_img)
        # try
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        # 转换图片格式
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            img = img.convert(conversion_format)
        # 图片转数组
        img = np.asarray(img)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            img = np.expand_dims(img, -1)
        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            img = img[:, :, ::-1]
        # handle grayscale mixed in RGB images
        elif len(img.shape) == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        # 数组转图片
        img = Image.fromarray(img)
        # 返回
        return img


def get_img_txt(img_path, txt_path, transforms):
    # 初始化
    img = []
    txt = []
    idx = {}
    num = 0
    # 加载图片
    logger.info("正在加载图片...")
    name_img = os.listdir(img_path)
    pbar = tqdm(total=100)
    for name in name_img:
        picture = read_img(os.path.join(img_path, name), "RGB")
        picture = transforms(picture)
        img.append(picture)
        idx[num] = name
        num += 1
        pbar.update(100 / len(name_img))
    pbar.close()
    logger.info("图片加载已完成~")
    # 加载文本
    logger.info("正在加载文本...")
    file_txt = open(txt_path, 'r')
    for line in file_txt.readlines():
        sentence = line.strip()
        sentence = sentence.split('$')[-1]
        txt.append(sentence)
    logger.info("文本加载已完成~")
    # 返回
    return img, txt, idx


def get_similarity(model, img, txt, device):
    # 汇报
    logger.info("正在计算相似度...")
    # 参数
    stage = 1000
    num_img = len(img)
    # 初始化
    img_feature_list = []
    txt_feature_list = []
    # 循环：数据加载+特征前传
    for i in range(0, num_img, stage):
        # 参数
        start = i
        end = min(num_img, i + stage)
        # 图片加载
        img_input = [item.cpu().detach().numpy() for item in img[start: end]]
        img_input = numpy.array(img_input)
        img_input = torch.tensor(img_input).cuda()
        # 文本加载
        txt_token = [tokenizer.encode("" + desc) for desc in txt[start: end]]
        txt_input = torch.zeros((len(txt_token), 77), dtype=torch.int64)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        for j, tokens in enumerate(txt_token):
            tokens = [sot_token] + tokens + [eot_token]
            txt_input[j, :len(tokens)] = torch.tensor(tokens)
        # 计算特征
        with torch.no_grad():
            # 准备
            img_in = img_input.clone().detach().to(device)
            txt_in = txt_input.clone().detach().to(device)
            # 前传
            img_feature = model.clip.encode_vis(img_in, args)
            txt_feature = model.clip.encode_seq(txt_in, args)
            img_feature_list.append(img_feature)
            txt_feature_list.append(txt_feature)
    # 调整
    img_feature = torch.cat(img_feature_list)
    txt_feature = torch.cat(txt_feature_list)
    # 规范化
    img_feature /= img_feature.norm(dim=-1, keepdim=True)  # [17611,512]
    txt_feature /= txt_feature.norm(dim=-1, keepdim=True)  # [17611,512]
    # 相似度计算
    similarity = torch.matmul(txt_feature, img_feature.t()).cpu().numpy()
    # 汇报
    logger.info("相似度计算已完成~")
    return similarity


def infer_to_json(similarity, idx, txt):
    # 调整
    similarity_argsort = np.argsort(-similarity, axis=1)
    # 准备
    top_k = 10
    result_list = []
    # 加载结果列表
    for i in range(len(similarity_argsort)):
        dictionary = {'text': txt[i], 'image_names': []}
        for j in range(top_k):
            dictionary['image_names'].append(idx[similarity_argsort[i, j]])
        result_list.append(dictionary)
    # 写入json文件
    with open('result_{}/{}_T_{}_epoch{}_mlp.json'.format(args.infer_mode, args.infer_mode, int(args.T), args.model_num), 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))
    logger.info("推理已完成~")


def main():
    # setting the testing device
    device, n_gpu = init_device(args, args.local_rank)
    # load dataloader
    if args.infer_mode == 'test':
        _, data_length = dataloader_baidu_infer(args, tokenizer)
        args.bank_size = data_length
    else:
        _, data_length = dataloader_baidu_infer(args, tokenizer)
        args.bank_size = data_length
    # init model
    model = init_model(args, device)
    # init val_test dataloader
    transforms = TransformsVal(224)
    # 加载数据
    img, txt, idx = get_img_txt(img_path, txt_path, transforms)
    # 计算相似度
    similarity = get_similarity(model, img, txt, device)
    # 推理结果
    infer_to_json(similarity, idx, txt)


if __name__ == "__main__":
    main()

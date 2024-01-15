import torch


def get_model_path(args):
    model_path = {  # pretrained resnet50-lr0.001
        'brightness': f'/home/yxue/TTA-try/checkpoint/ckpt_[\'brightness\']_[{args.severity}].pt',
        'defocus_blur': f'/home/yxue/TTA-try/checkpoint/ckpt_[\'defocus_blur\']_[{args.severity}].pt',
        'gaussian_noise': f'/home/yxue/TTA-try/checkpoint/ckpt_[\'gaussian_noise\']_[{args.severity}].pt',
        'jpeg_compression': f'/home/yxue/TTA-try/checkpoint/ckpt_[\'jpeg_compression\']_[{args.severity}].pt',
        'snow': f'/home/yxue/TTA-try/checkpoint/ckpt_[\'snow\']_[{args.severity}].pt',
    }

    param_ls = []
    for d in model_path:
        ttt = torch.load(model_path[d], map_location='cpu')
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['model'])
    return param_ls
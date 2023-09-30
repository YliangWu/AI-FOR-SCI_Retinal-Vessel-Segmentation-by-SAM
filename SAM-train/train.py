# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
# %% setup environment
import matplotlib.pyplot as plt
import os
import monai
from segment_anything import sam_model_registry
from datetime import datetime

import shutil
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import re
import os
from utils.BVdata import *
import glob
from metrics.metric import *
from model.BVSam import BVSAM

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
join = os.path.join
# torch.distributed.init_process_group(backend="gloo")
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def main(args):

    # %% set up model for training
    # device = args.device
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)

    img_path = args.image
    mask_path = args.mask_path
    pixel_mean = [0.5]*3
    pixel_std = [0.5]*3

    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    basename = os.path.basename(img_path)
    _, ext = os.path.splitext(basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        img_paths = [file for file in glob.glob(os.path.join(img_path, "*.*")) if regex.match(file)]
        print("train with {} imgs".format(len(img_paths)))
        mask_paths = [os.path.join(mask_path, os.path.basename(file).replace(".jpg", '_1stHO.png')) for file in
                      img_paths]
    else:
        bs = 1
        img_paths = [img_path]
        mask_paths = [mask_path]
        num_workers = 1
    metric = Metric(num_classes=1)

    #注册一个sam
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    #原封不动传入
    bvsam_model = BVSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    bvsam_model.train()
    img_mask_encdec_params = list(bvsam_model.image_encoder.parameters()) + list(
        bvsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10  #初始最大loss
    train_dataset = SegDataset(img_paths=img_paths, mask_paths=mask_paths, mask_divide=True, divide_value=255,
                               pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=1024)
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            bvsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0  #每个epoch的平均loss
        final_step=0
        for step, (image, target) in enumerate(train_dataloader):
            final_step=step
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    pred = bvsam_model(image)
                    loss = seg_loss(pred, target) + ce_loss(
                        pred, target.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                pred = bvsam_model(image)
                loss = seg_loss(pred, target) + ce_loss(pred, target.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            metric.update(torch.softmax(pred, dim=1), target)
            print('Time: {},epoch:{}-{}: loss:{}'.format(datetime.now().strftime("%Y%m%d-%H%M"),epoch+1, step+1, loss.item()))
            epoch_loss += loss.item()

        epoch_loss /= (final_step+1)  #该epoch内平均loss
        losses.append(epoch_loss)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        iou = np.nanmean(metric.evaluate()["iou"][1:].numpy())
        print("epoch-{}: iou:{}".format(epoch, iou.item()))
        # save the best model for each 10 epochs
        if epoch_loss < best_loss and epoch%10==0:
            best_loss = epoch_loss
            checkpoint = {
                "model": bvsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }


    ## save the lastest model
    checkpoint = {
        "model": bvsam_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, join(model_save_path, args.save_path))

    # %% plot loss
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
    plt.close()


if __name__ == "__main__":
    # %% set up parser
    parser = argparse.ArgumentParser("Learnable prompt")
    parser.add_argument("--image", type=str, default='./dataset/img',
                        help="path to the image that used to train the model")
    parser.add_argument("--mask_path", type=str, default='./dataset/mask',
                        help="path to the mask file for training")
    parser.add_argument("-task_name", type=str, default="SEG_SAM")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth",
                        help="model's original weight's path")
    parser.add_argument("--save_path",type=str,default="BVsam_model_latest.pth",
                        help="the path to save the model's weight")

    # parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    # train
    parser.add_argument("-num_epochs", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-num_workers", type=int, default=0)
    # Optimizer parameters
    parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("-lr", type=float, default=0.001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()


    main(args)

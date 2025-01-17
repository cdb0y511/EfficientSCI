import os
import os.path as osp
import sys
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_single_image,get_device_info,load_checkpoints
from cacti.utils.metrics import compare_psnr,compare_ssim
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import time
import einops
import lpips



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',config_name)
    data_name_list = os.listdir(cfg.test_data.data_root)
    mask_list,mask_list_s,cr_list = [],[],[]
    for name in data_name_list:
        mask,mask_s = generate_masks(cfg.test_data.mask_path+'/'+name,cfg.test_data.mask_shape)
        mask_list.append(mask)
        mask_list_s.append(mask_s)
        cr = mask.shape[0]
        cr_list.append(cr)
    if args.weights is None:
        args.weights = cfg.checkpoints

    test_dir = osp.join(args.work_dir,"test_images")

    log_dir = osp.join(args.work_dir,"test_log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line)
    test_data = build_dataset(cfg.test_data,{"mask_list":mask_list})
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(args.weights)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

    psnr_dict,ssim_dict,lpips_dic = {},{},{}
    psnr_list,ssim_list,lpips_list = [],[],[]
    lpips_model = lpips.LPIPS(net="alex")

    sum_time=0.0
    time_count = 0

    for data_iter,data in enumerate(data_loader):
        Phi = einops.repeat(mask_list[data_iter], 'cr h w->b cr h w', b=1)
        Phi_s = einops.repeat(mask_list_s[data_iter], 'h w->b 1 h w', b=1)
        Phi = torch.from_numpy(Phi).to(args.device)
        Phi_s = torch.from_numpy(Phi_s).to(args.device)

        psnr,ssim,lpips_value = 0,0,0
        batch_output = []
        gt = None
        try:
            meas, gt = data #meas shape: [c h w] gt shape: [c cr h w]
        except:
            meas = data
        if gt is not None:
            gt = gt[0].numpy()
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]
        name = test_data.data_name_list[data_iter]
        if "_" in name:
            _name,_ = name.split("_")
        else:
            _name,_ = name.split(".")
        for ii in range(batch_size):
            if gt is not None:
                single_gt = gt[ii]
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                if "amp" in cfg.keys() and cfg.amp:
                    with autocast():
                        outputs = model(single_meas, Phi, Phi_s)
                else:
                    outputs = model(single_meas, Phi, Phi_s)
                torch.cuda.synchronize()
                end = time.time()
                run_time = end - start
                if ii>0:
                    sum_time += run_time
                    time_count += 1
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy().astype(np.float32)

            batch_output.append(output)
            if gt is not None:
                for jj in range(cr_list[data_iter]):
                    per_frame_out = output[jj]
                    per_frame_gt = single_gt[jj]
                    psnr += compare_psnr(per_frame_gt,per_frame_out*255)
                    ssim += compare_ssim(per_frame_gt,per_frame_out*255)
                    tmp = lpips_model(torch.tensor(per_frame_gt/255).unsqueeze(0).float(), torch.tensor(per_frame_out).unsqueeze(0).float()).detach().cpu().numpy()
                    lpips_value += tmp[0][0][0][0]

        psnr = psnr / (3* cr_list[data_iter])
        ssim = ssim / (3* cr_list[data_iter])
        lpips_value = lpips_value / (3* cr_list[data_iter])
        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f} Mean LPIPS {:.4f}.".format(
                    _name,psnr,ssim,lpips_value))
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_value)
        psnr_dict[_name] = psnr
        ssim_dict[_name] = ssim
        lpips_dic[_name] = lpips_value
        #save image
        out = np.array(batch_output)
        image_dir = osp.join(test_dir,_name)
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        save_single_image(out,image_dir,0,name=config_name,demosaic = False, combineRGB = True)
    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    lpips_dic["lpips_mean"] = np.mean(lpips_list)
    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    lpips_str = ", ".join([key+": "+"{:.4f}".format(lpips_dic[key]) for key in lpips_dic.keys()])
    logger.info("Mean PSNR: \n"+
                dash_line + 
                "{}.\n".format(psnr_str)+
                dash_line)

    logger.info("Mean SSIM: \n"+
                dash_line + 
                "{}.\n".format(ssim_str)+
                dash_line)
    logger.info("Mean LPIPS: \n"+
                dash_line +
                "{}.\n".format(lpips_str)+
                dash_line)

if __name__=="__main__":
    main()
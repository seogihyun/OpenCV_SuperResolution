import cv2
import torch
import argparse
import numpy as np
import time
import lpips as lp
from models import FSRCNN_x
from utils import preprocess, calc_psnr, calc_ssim, LPIPS


def writeVideo():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        cap = cv2.VideoCapture('./car.mp4')
    except:
        print('Try again!')
        return

    fps = 20.0
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    """ 비디오 저장 """
    out = cv2.VideoWriter('{name}_{model}_{title}.{extension}'.format(name='Gihyun', model='FSRCNN-x', title='car', extension='mp4'), fourcc, fps, (width,height))

    model = FSRCNN_x(scale_factor=args.scale).to(device)

    
    try:
        model.load_state_dict(torch.load(args.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)


    model.eval()

    lpips_metric = lp.LPIPS(net='vgg')

    while True:
        
        start = time.time()
        
        """ 재생되는 비디오의 한 프레임씩 읽어옴 """
        ret, frame = cap.read()

        if ret:

            hr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            lr = cv2.resize(hr, (width // args.scale, height // args.scale), interpolation=cv2.INTER_CUBIC)

            # bicubic upscaling
            bicubic = cv2.resize(lr, (width ,height), interpolation=cv2.INTER_CUBIC)


            # ycbcr값
            lr = preprocess(lr, device)
            hr = preprocess(hr, device)

            with torch.no_grad():
                preds = model(lr).clamp(0.0, 1.0)


            """ PSNR,SSIM, LPIPS 계산 """
            # # PSNR
            # psnr = calc_psnr(hr, preds)
            # print('PSNR: {:.2f}'.format(psnr))
            # # SSIM
            # ssim = calc_ssim(hr, preds)
            # print('SSIM: {:.2f}'.format(ssim))
            # # LPIPS (Learned Perceptual Image Patch Similarity)
            # lpips = LPIPS(hr.cpu(), preds.cpu(), lpips_metric)
            # print('LPIPS: {:.2f}'.format(lpips))


            preds = preds.mul(255.0).cpu().numpy().squeeze(0)

            """ output : (c,h,w) -> (h,w,c)로 변경 """
            sr_image = np.array(preds).transpose([1, 2, 0])
            sr_image = np.clip(sr_image, 0.0, 255.0).astype(np.uint8)


            """ 이미지 중앙 부분 절반씩 잘라서 붙이기"""
            bicubic = bicubic[:, 480:1440]
            sr_image = sr_image[:, 480:1440]

            # print('bicubic : ', bicubic)
            # print('sr_image : ', sr_image)

            # Window name in which image is displayed 
            bicubic_name = 'Bicubic'
            SR_name = 'Gihyun'
            # org (글자 위치)
            org = (800, 50) 
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX  
            # fontScale (글자 크기)
            fontScale = 1
            # Blue color in BGR 
            color = (0, 0, 0) 
            # Line thickness of 2 px (두께) 
            thickness = 2
            # Using cv2.putText() method 
            bicubic = cv2.putText(bicubic, bicubic_name, org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Using cv2.putText() method 
            sr_image = cv2.putText(sr_image, SR_name, org, font, fontScale, color, thickness, cv2.LINE_AA)
            sr_image = sr_image.get()


            output = cv2.hconcat([bicubic, sr_image])

            # cv2.imshow('output', output)
                
            out.write(output)

            end = time.time()
            print('time elapsed : {:.2f}'.format(end - start))

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--video_file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    writeVideo()


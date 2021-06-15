
# Video capture and writer using OpenCV with FSRCNN-x


<center><img src="https://user-images.githubusercontent.com/72849922/122023411-79950d80-ce02-11eb-88b7-9b34aa37b311.PNG"></center>

-----


## Prepare

- train_x2.h5

```bash
python prepare.py --images_dir "./train_file" \
                  --h5_dir "./h5_dir/train/train_x2.h5" \
                  --scale 2
```

- eval_x2.h5

```bash
python prepare.py --images-dir "./eval_file" \
                  --h5_dir "./h5_dir/eval/eval_x2.h5" \
                  --scale 2 \
                  --eval
```

## Train


```bash
python train.py --train_file "./h5_dir/train/train_x2.h5" \
                --eval_file "./h5_dir/eval/eval_x2.h5" \
                --weights_dir "./weights_dir" \
                --scale 3               
```

## Video

```bash
python save_video.py --weights_file "./weights/x2/best.pth" \
                     --video_file "./video_file/video.mp4" \
                     --model_name "FSRCNN-x" \
                     --scale 2
```


## Video with PSNR, SSIM, LPIPS average check

```bash
python save_video.py --weights_file "./weights/x2/best.pth" \
                     --video_file "./video_file/video.mp4" \
                     --model_name "FSRCNN-x" \
                     --scale 2
                     --calc_check
```

## Result

![opencv_fsrcnn-x](https://user-images.githubusercontent.com/72849922/122024431-62a2eb00-ce03-11eb-9c3b-86006760cc3b.gif)




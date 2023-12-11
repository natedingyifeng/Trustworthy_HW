# Improving Automatic Surgical Fine-Tuning with Spectrum-based Fault Localization

## Environment

Create an environment with the following command:
```
conda env create -f conda_env.yml
```

## Running CRGN Surgical Fine-tuning
```
python main.py --config-name='cifar-10c' args.train_n=1000 args.seed=0 data.corruption_types=[frost,gaussian_blur,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur]  wandb.use=False args.auto_tune=RGN-with-pass-tarantula args.epochs=15

python main.py --config-name='imagenet-c' args.train_n=5000 args.seed=2 data.corruption_types=[brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur] wandb.use=False args.auto_tune=RGN-with-pass-wong2-version-2 args.epochs=10

```


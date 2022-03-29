
# Some experiments comparing CSGM and JPMAP (add "--save-iters" option to store JPMAP iterations)

# Inpainting (random mask)
python run_algorithms_hdn.py --dataset celeba-wb --model celeba_dvae7 --model_noise 7 --problem missingpixels --sigma 10 --missing 0.80 --n-samples 3 --iters-jpmap 300 --exp-name missing_hdn --save-iters --uzawa --alpha 1  # Exponential multiplier method
python run_algorithms_hdn.py --dataset celeba-wb --model celeba_dvae7 --model_noise 7 --problem missingpixels --sigma 10 --missing 0.80 --n-samples 3 --iters-jpmap 300 --exp-name missing_hdn --save-iters  # beta=1e-3/gamma^2

# Denoising
python run_algorithms_hdn.py --dataset celeba-wb --model celeba_dvae7 --model_noise 7 --problem denoising --sigma 110 --n-samples 3 --iters-jpmap 300 --exp-name denoising_hdn --save-iters --uzawa --alpha 1  # Exponential multiplier method
python run_algorithms_hdn.py --dataset celeba-wb --model celeba_dvae7 --model_noise 7 --problem denoising --sigma 110 --n-samples 3 --iters-jpmap 300 --exp-name denoising_hdn --save-iters  # beta=1e-3/gamma^2

# Compressed Sensing
# never tested but implemented, shall work with a bit of tuning
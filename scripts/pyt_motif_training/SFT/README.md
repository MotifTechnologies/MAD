# How to run

## With activation functions 
```
cd MAD/scripts/pyt_motif_training/SFT
sky launch -c <your_exp_name> ./<gpu_version>/sky.yaml
```

## With optimized activation functions 
```
cd MAD/scripts/pyt_motif_training/SFT
sky launch -c <your_exp_name> ./<gpu_version>/sky_kernel.yaml
```
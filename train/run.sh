# CUDA_VISIBLE_DEVICES=9 python demo.py \
#     --task img2img \
#     --data_path './assets_demo_cli/' \
#     --data_items dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557 \
#     --num_inputs 1 \
#     --chunk_strategy nearest-gt \
#     --video_save_fps 10

# NAME=dbg && CUDA_VISIBLE_DEVICES=6,7,8,9 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
NAME=dbg && CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m train.trainval \
    --output_dir ./work_dirs/${NAME} \
    --no_use_torch_compile \
    --visual_every 100 \
    --print_every 100 \
    --test_every -1 \
    --amp --amp_dtype fp16 \
    --batch_scenes 1

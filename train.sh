source /data/xkong016/miniconda3/etc/profile.d/conda.sh
conda activate mutual_infomation

export CUDA_VISIBLE_DEVICES=0
python main.py  --strength 200 --task_type multinormal  --dim 3
python main.py  --strength 200 --task_type spiral  --dim 3  
python main.py  --strength 200 --task_type half_cube  --dim 3
#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=64                                # CPU cores/threads
#SBATCH --gres=gpu:1       	                              # Number of GPUs (per node)
#SBATCH --mem 96000        	                              # Reserve 96 GB RAM for the job
#SBATCH --time 1-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name fig-seg-model                        # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-seg-model-train.out     # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-seg-model-train.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy                                   # run only on dizzy

# activate shell
source /home/kapilg/.local/share/virtualenvs/every-part-matters-segmenter-a9DYQe8H/bin/activate

# go to the correct directory
cd /home/kapilg/projects/every-part-matters-segmenter

# run code
python train.py \
--dataset figure_seg \
--dataset_dir ./assets/dataset \
--figure_seg_data train \
--neg_sample_rate 1 \
--combine_sample_rate 1 \
--base_model liuhaotian/llava-v1.6-vicuna-7b \
--pretrain_mm_mlp_adapter ./assets/llava-v1.6-vicuna-7b/mm_projector.bin \
--vision-tower ./assets/clip-vit-large-patch14 \
--vision_pretrained ./assets/vision-pretrained/sam_vit_h_4b8939.pth


# deactivate environment
exit
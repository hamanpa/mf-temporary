#!/bin/bash

#SBATCH --error=logs/inspection_%j_%a.log
#SBATCH --output=logs/inspection_%j_%a.log
#SBATCH --job-name=inspection
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --mem=62G
#SBATCH --time=168:00:00
#SBATCH --exclude=w[9,11,13-17]
#SBATCH --cpus-per-task=16

# Map array index to a specific parameter config
case $SLURM_ARRAY_TASK_ID in
    0) 
        PARAM="stimulus.drive_rate"
        VALUES="[0.5, 1.0, 1.5, 2.0]"
        STIM="SpontActivity"
        ;;
    1) 
        PARAM="network.neurons.exc_neuron.neuron_params.b"
        VALUES="[0.02, 0.05, 0.08, 0.11]"
        STIM="SpontActivity"
        ;;
    2) 
        PARAM="network.synapses.exc_neuron.syn_params.tau_rec"
        VALUES="[30.0, 80.0, 130.0, 180.0]"
        STIM="SpontActivity"
        ;;
esac

# Execute the python script
source /home/haman/virt_env/mf-csng/bin/activate

python run_inspection.py --param "$PARAM" --values "$VALUES" --stim "$STIM"
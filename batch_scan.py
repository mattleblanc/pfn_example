"""
 batch_scan.py

 This script is for testing the Brown CCV batch sytems.
 It will train a set of PFNs and scan over the dimension of the latent space.

 Matt LeBlanc 2024
"""

import os

latent_sizes = [2,4,8,16,32,64,128,256,512]

for l in latent_sizes:
    with open('batch/train_'+str(l)+'.sh', 'w') as run_script:

        run_script.write('#!/bin/bash\n\n')
        
        run_script.write('#SBATCH -N 1\n')
        run_script.write('#SBATCH -n 1\n')
        run_script.write('#SBATCH --mem=32G\n')
        run_script.write('#SBATCH -t 4:00:00\n')

        # https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu
        run_script.write('#SBATCH -p gpu --gres=gpu:1\n')
        
        run_script.write('#SBATCH -o logs/slurm-%j.out\n')
        run_script.write('#SBATCH -J PFN-l'+str(l)+'\n')
        run_script.write('\n')

        run_script.write('source tensorflow.venv/bin/activate\n\n')
        
        cmd =  'python pfn_train.py '
        cmd += ' --doEarlyStopping'
        cmd += ' --latentSize='+str(l)
        cmd += ' --makeROCs'    
        cmd += ' --label=\"l'+str(l)+'\" '
        
        print(cmd)
        
        run_script.write(cmd+'\n')

    print('sbatch batch/train_'+str(l)+'.sh')
    os.system('sbatch batch/train_'+str(l)+'.sh')

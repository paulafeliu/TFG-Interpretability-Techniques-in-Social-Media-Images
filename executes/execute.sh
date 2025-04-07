#!/bin/bash
#SBATCH -n 2                      # Número de núcleos
#SBATCH -N 1                      # Asegura que todos los núcleos estén en la misma máquina
#SBATCH -D /fhome/pfeliu/tfg_feliu  # Directorio de trabajo (ajústalo)
#SBATCH -t 0-02:00                # Tiempo de ejecución máximo (D-HH:MM), ajústalo según tus necesidades
#SBATCH -p tfg                    # Cola/partition (puedes cambiar a tfgm si lo requiere)
#SBATCH --mem 4096                # Memoria solicitada (en MB)
#SBATCH -o %j.out           # Archivo donde se guarda STDOUT
#SBATCH -e %j.err           # Archivo donde se guarda STDERR
#SBATCH --gres gpu:1              # Solicita 1 GPU

#module load python/3.8           

source /fhome/pfeliu/env_tfg/bin/activate

python3 /fhome/pfeliu/tfg_feliu/code/twitter_analysis.py
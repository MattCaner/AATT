#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J test_cuda
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=1GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgwfisdiplom21csgpu
## Specyfikacja partycji
#SBATCH -p plgrid-gpu
## Plik ze standardowym wyjściem
#SBATCH --output=$SCRATCH/output.out
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR/aatt

srun /bin/hostname
module load test/pytorch/1.1.0
module load plgrid/apps/cuda/11.3
python test_cuda.py
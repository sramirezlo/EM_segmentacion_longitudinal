#!/usr/bin/env python3
"""
Entrenamientos Personalizados nnU-Net v2 - TFM Detección Lesiones Esclerosis Múltiple
===================================================================================
Fase 03: 4 entrenamientos Dataset202 (20/50/100/250 épocas) + verificación preprocesado

Entrenamientos:
- nnUNetTrainer_20epochs # este entrenamiento ya existe en nnU-Net v2, no hace falta definirlo (viene de base)
- nnUNetTrainer_50epochs (custom)
- nnUNetTrainer_100epochs (custom)
- nnUNetTrainer_250epochs (custom)

Resultados en: data/nnunet_results/Dataset202_ImaginEM_FLAIR_HOLDOUT/
"""

import os
import sys
from pathlib import Path
import nnunetv2

# Configuración matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Se definen las variables de entorno nnU-Net v2
NNUNET_RAW = Path("data/nnunet_raw")
NNUNET_PREPROCESSED = Path("data/nnunet_preprocessed")
NNUNET_RESULTS = Path("data/nnunet_results")

os.environ['nnUNet_raw'] = str(NNUNET_RAW)
os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
os.environ['nnUNet_results'] = str(NNUNET_RESULTS)

DATASET_ID = 202
CONFIG = "3d_fullres"
FOLD = 0

print("Variables de entorno:")
for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
    print(f"  {var}: {os.environ[var]}")

# ============================================================================
# VERIFICACIÓN PREPROCESADO
# ============================================================================

def verificar_preprocesado():
    """Verifica que el preprocesado de Dataset202 existe."""
    print("\n" + "="*80)
    print("1. VERIFICANDO PREPROCESADO Dataset202")
    print("="*80)

    preprocessed_path = NNUNET_PREPROCESSED / f"Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT"

    if preprocessed_path.exists():
        print(f" Preprocesado encontrado: {preprocessed_path}")
        configs = [p.name for p in preprocessed_path.glob("nnUNetPlans*")]
        print(f"   Configuraciones: {configs}")

        # Contar folds
        for config_dir in configs:
            folds = list((preprocessed_path / config_dir).glob("*fold*"))
            print(f"   {config_dir}: {len(folds)} folds")
    else:
        print(f" ERROR: Preprocesado NO encontrado en {preprocessed_path}")
        print("Ejecuta primero: python 02_holdout_preprocesado.py")
        sys.exit(1)

# ============================================================================
# SE CREAN LOS TRAINERS PERSONALIZADOS, con ayuda del repositorio GitHub https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py
# ============================================================================

def crear_trainers_personalizados():
    """Crea trainers nnUNetTrainer_Xepochs en el paquete nnU-Net."""
    print("\n" + "="*80)
    print("2. CREANDO TRAINERS PERSONALIZADOS (50/100/250 épocas)")
    print("="*80)

    # Detectar instalación nnU-Net
    nnunet_root = Path(nnunetv2.__file__).parent
    print(f"nnU-Net instalado en: {nnunet_root}")

    # Directorio trainers personalizados
    training_length_dir = nnunet_root / "training" / "nnUNetTrainer" / "variants" / "training_length"
    training_length_dir.mkdir(parents=True, exist_ok=True)

    trainer_path = training_length_dir / "custom_epochs.py"

    trainer_code = '''import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_50epochs(nnUNetTrainer):
    """Trainer nnU-Net personalizado: 50 épocas."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        print(f" Trainer 50epochs inicializado para Dataset{plans['dataset_name']}")


class nnUNetTrainer_100epochs(nnUNetTrainer):
    """Trainer nnU-Net personalizado: 100 épocas."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        print(f" Trainer 100epochs inicializado para Dataset{plans['dataset_name']}")


class nnUNetTrainer_250epochs(nnUNetTrainer):
    """Trainer nnU-Net personalizado: 250 épocas."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        print(f" Trainer 250epochs inicializado para Dataset{plans['dataset_name']}")
'''

    with open(trainer_path, "w") as f:
        f.write(trainer_code)

    print(f" Trainers creados: {trainer_path}")
    print("   - nnUNetTrainer_50epochs")
    print("   - nnUNetTrainer_100epochs")
    print("   - nnUNetTrainer_250epochs")

# ============================================================================
# COMANDOS DE ENTRENAMIENTO
# ============================================================================
# Se define una función que ejecuta los 4 entrenamientos
def generar_comandos_entrenamiento():
    """Genera comandos nnUNetv2_train para los 4 entrenamientos."""
    print("\n" + "="*80)
    print("3. COMANDOS DE ENTRENAMIENTO (ejecutar manualmente)")
    print("="*80)

    entrenamientos = [
        ("20epochs", "nnUNetTrainer", "De serie en nnU-Net v2"), # preestablecido en nnU-Net v2 (configurado de serie)
        ("50epochs", "nnUNetTrainer_50epochs", "Custom 50 épocas"),
        ("100epochs", "nnUNetTrainer_100epochs", "Custom 100 épocas"),
        ("250epochs", "nnUNetTrainer_250epochs", "Custom 250 épocas")
    ]

    comandos = []
    for epochs, trainer, desc in entrenamientos:
        cmd = f"nnUNetv2_train {DATASET_ID} {CONFIG} {FOLD} -tr {trainer} --npz"
        comandos.append((epochs, cmd, desc))

        print(f"\n {desc}:")
        print(f"   {cmd}")
        print(f"   → results: {NNUNET_RESULTS}/Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT/nnUNetTrainer__{CONFIG}__{trainer}__{FOLD}/")

    return comandos

# ============================================================================
# VERIFICACIÓN RESULTADOS
# ============================================================================

def verificar_resultados():
    """Verifica entrenamientos completados."""
    print("\n" + "="*80)
    print("4. VERIFICANDO RESULTADOS ENTRENAMIENTOS")
    print("="*80)

    results_base = NNUNET_RESULTS / f"Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT"

    if results_base.exists():
        trainers = [d for d in results_base.iterdir() if d.is_dir()]
        print(f" Results base: {results_base}")
        print(f"   Entrenamientos encontrados: {len(trainers)}")

        for trainer_dir in sorted(trainers):
            fold_dirs = [d for d in trainer_dir.iterdir() if d.is_dir()]
            print(f"   {trainer_dir.name}: {len(fold_dirs)} folds")
    else:
        print(f" Results esperado: {results_base}")
        print("Ejecuta los comandos de entrenamiento primero")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(" Fase 03: Entrenamientos Personalizados nnU-Net")

    verificar_preprocesado()
    crear_trainers_personalizados()
    comandos = generar_comandos_entrenamiento()
    verificar_resultados()

    print("\n" + " SCRIPT COMPLETADO!")
    print("\n PASOS PARA ENTRENAR:")
    for epochs, cmd, _ in comandos:
        print(f"1. {cmd}")

    print(f"\n Resultados finales en: {NNUNET_RESULTS}/Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT/")

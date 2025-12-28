#!/usr/bin/env python3
"""
Predicciones y Evaluación Holdout - TFM Detección Lesiones Esclerosis Múltiple
============================================================================
Fase 04: Predicciones 4 modelos en ImaginEM test (15%) + MSSEG2 (40 casos)

Datasets de test:
- Dataset210_ImaginEM_FLAIR_HOLDOUT_TEST (ImaginEM test holdout)
- Dataset302_MSSEG2_FLAIR (MSSEG2 completo)

Modelos: 20/50/100/250 épocas --> 4 predicciones en ImaginEM y 4 predicciones en MSSEG2
"""

import os
import json
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import zipfile

# Configuración matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NNUNET_RAW = Path("data/nnunet_raw")
NNUNET_RESULTS = Path("data/nnunet_results")
MSSEG2_ZIP = Path("data/raw/MSSEG-2/LongitudinalMultipleSclerosisLesionSegmentationChallengeMiccai21_v2.zip")

# Variables de entorno
os.environ['nnUNet_raw'] = str(NNUNET_RAW)
os.environ['nnUNet_results'] = str(NNUNET_RESULTS)

DATASET_ID = 202
CONFIG = "3d_fullres"
FOLD = 0
CHECKPOINT = "checkpoint_best.pth"

print("Variables configuradas:")
print(f"  nnUNet_raw: {NNUNET_RAW}")
print(f"  nnUNet_results: {NNUNET_RESULTS}")

# ============================================================================
# 1. VERIFICACIONES INICIALES
# ============================================================================

def verificar_prerequisitos():
    """Verifica datasets y entrenamientos previos."""
    print("\n" + "="*80)
    print("1. VERIFICANDO PRERREQUISITOS")
    print("="*80)

    # Dataset102 con split
    dataset102 = NNUNET_RAW / "Dataset102_ImaginEM_FLAIR"
    assert dataset102.exists(), f" Necesario: python 02_holdout_preprocesado.py"
    assert (dataset102 / "split_holdout.json").exists()

    # Entrenamientos completados
    results_base = NNUNET_RESULTS / f"Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT"
    trainers = ["nnUNetTrainer", "nnUNetTrainer_50epochs", "nnUNetTrainer_100epochs", "nnUNetTrainer_250epochs"]

    for trainer in trainers:
        trainer_dir = results_base / f"{trainer}_nnUNetPlans_{CONFIG}"
        assert trainer_dir.exists(), f" Entrenamiento faltante: {trainer_dir}"
        assert (trainer_dir / f"fold_{FOLD}" / CHECKPOINT).exists()

    print(" Todos los prerrequisitos OK")

# ============================================================================
# 2. DATASET210_IMAGINEM_TEST (15% HOLDOUT)
# ============================================================================

def crear_dataset210_imaginem_test():
    """Crea Dataset210 con test holdout de ImaginEM."""
    print("\n" + "="*80)
    print("2. CREANDO DATASET210_IMAGINEM_FLAIR_HOLDOUT_TEST")
    print("="*80)

    dataset102 = NNUNET_RAW / "Dataset102_ImaginEM_FLAIR"
    split_path = dataset102 / "split_holdout.json"

    with open(split_path, "r") as f:
        split = json.load(f)

    test_cases = split["test"]
    print(f"Test cases ImaginEM: {len(test_cases)}")

    # Crear dataset test
    dataset210 = NNUNET_RAW / "Dataset210_ImaginEM_FLAIR_HOLDOUT_TEST"
    imagesTs = dataset210 / "imagesTs"
    labelsTs = dataset210 / "labelsTs"

    for d in [imagesTs, labelsTs]:
        d.mkdir(parents=True, exist_ok=True)

    imagesTr_src = dataset102 / "imagesTr"
    labelsTr_src = dataset102 / "labelsTr"

    for case_id in test_cases:
        # Imágenes (2 canales)
        for ch in ["0000", "0001"]:
            src = imagesTr_src / f"{case_id}_{ch}.nii.gz"
            dst = imagesTs / src.name
            shutil.copy(src, dst)

        # Labels
        src_lbl = labelsTr_src / f"{case_id}.nii.gz"
        dst_lbl = labelsTs / src_lbl.name
        shutil.copy(src_lbl, dst_lbl)

    # dataset.json referencia
    shutil.copy(dataset102 / "dataset.json", dataset210 / "dataset.json")

    print(f" Dataset210 creado: {dataset210}")
    print(f"   imagesTs: {len(list(imagesTs.glob('*.nii.gz')))}")
    print(f"   labelsTs: {len(list(labelsTs.glob('*.nii.gz')))}")

    return dataset210

# ============================================================================
# 3. DATASET302_MSSEG2 (40 CASOS)
# ============================================================================

def crear_dataset302_msseg2():
    """Crea Dataset302_MSSEG2_FLAIR desde ZIP."""
    print("\n" + "="*80)
    print("3. CREANDO DATASET302_MSSEG2_FLAIR")
    print("="*80)

    # Extraer ZIP si no existe
    extract_dir = Path("data/raw/MSSEG-2/LongitudinalMultipleSclerosisLesionSegmentationChallengeMiccai21_v2")
    if MSSEG2_ZIP.exists() and not extract_dir.exists():
        print(f" Extrayendo {MSSEG2_ZIP}...")
        with zipfile.ZipFile(MSSEG2_ZIP, 'r') as z:
            z.extractall(extract_dir.parent)

    imgs_root = extract_dir / "training"
    assert imgs_root.exists(), f" MSSEG2 no encontrado en {imgs_root}"

    dataset302 = NNUNET_RAW / "Dataset302_MSSEG2_FLAIR"
    imagesTs = dataset302 / "imagesTs"
    labelsTs = dataset302 / "labelsTs"

    for d in [imagesTs, labelsTs]:
        d.mkdir(parents=True, exist_ok=True)

    cases = sorted([p for p in imgs_root.iterdir() if p.is_dir()])
    print(f"MSSEG2 casos: {len(cases)}")

    for cid in cases:
        case_dir = imgs_root / cid
        baseline = case_dir / "flair_time01_on_middle_space.nii.gz"
        followup = case_dir / "flair_time02_on_middle_space.nii.gz"
        gt = case_dir / "ground_truth.nii.gz"

        case_id = f"MSSEG2_{cid.name}"

        shutil.copy(baseline, imagesTs / f"{case_id}_0000.nii.gz")
        shutil.copy(followup, imagesTs / f"{case_id}_0001.nii.gz")
        shutil.copy(gt, labelsTs / f"{case_id}.nii.gz")

    # dataset.json específico MSSEG2 (labels 0,1)
    dataset_json = {
        "channel_names": {"0": "FLAIR_baseline", "1": "FLAIR_followup"},
        "labels": {"background": 0, "lesion_new": 1},
        "numTraining": 0,
        "file_ending": ".nii.gz"
    }
    with open(dataset302 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f" Dataset302 creado: {dataset302}")
    print(f"   imagesTs: {len(list(imagesTs.glob('*.nii.gz')))}")

    return dataset302

# ============================================================================
# 4. COMANDOS PREDICCIÓN
# ============================================================================

def generar_comandos_prediccion():
    """Genera comandos nnUNetv2_predict para 8 casos."""
    print("\n" + "="*80)
    print("4. COMANDOS PREDICCIÓN (8 TOTALES)")
    print("="*80)

    datasets = {
        "IMAGINEM_TEST": "Dataset210_ImaginEM_FLAIR_HOLDOUT_TEST/imagesTs",
        "MSSEG2": "Dataset302_MSSEG2_FLAIR/imagesTs"
    }

    trainers = [
        ("20ep", "nnUNetTrainer"),
        ("50ep", "nnUNetTrainer_50epochs"),
        ("100ep", "nnUNetTrainer_100epochs"),
        ("250ep", "nnUNetTrainer_250epochs")
    ]

    comandos = []
    results_base = NNUNET_RESULTS / f"Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT"

    for dataset_name, images_path in datasets.items():
        for epochs, trainer in trainers:
            output_dir = results_base / f"{dataset_name.lower()}_preds_{epochs}"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = f"""nnUNetv2_predict \\
  -i "{NNUNET_RAW}/{images_path}" \\
  -o "{output_dir}" \\
  -d {DATASET_ID} \\
  -c {CONFIG} \\
  -tr {trainer} \\
  -f {FOLD} \\
  -chk {CHECKPOINT}"""

            print(f"\n {dataset_name} - {epochs}:")
            print(f"   {cmd}")
            comandos.append((dataset_name, epochs, cmd))

    return comandos

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(" Fase 04: Predicciones ImaginEM Test + MSSEG2")

    verificar_prerequisitos()
    dataset210 = crear_dataset210_imaginem_test() # ImaginEM
    dataset302 = crear_dataset302_msseg2() # MSSEG2
    comandos = generar_comandos_prediccion()

    print("\n" + " DATASETS DE TEST CREADOS!")
    print("\n EJECUTA ESTOS 8 COMANDOS:")
    for dataset, epochs, cmd in comandos:
        print(f"\n{cmd}")

    print(f"\n Predicciones en: {NNUNET_RESULTS}/Dataset{DATASET_ID}_ImaginEM_FLAIR_HOLDOUT/")
    print("   - imagin_em_test_preds_20ep/50ep/100ep/250ep/")
    print("   - msseg2_preds_20ep/50ep/100ep/250ep/")

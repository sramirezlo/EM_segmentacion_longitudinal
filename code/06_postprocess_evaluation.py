#!/usr/bin/env python3
"""
Postprocesado Completo - TFM Detección Lesiones Esclerosis Múltiple
===================================================================
Fase 06: Filtros Volumen (P1,P3,P5,P10) + PH (k=3) + Combinaciones

Postprocesados aplicados al modelo 100 epochs:
1. Volumen P1, P3, P5, P10
2. PH k=3 (H0 bars >=3 en FLAIR followup bbox)
3. PH k=3 + Vol P3
4. Vol P3 + PH k=3

Evaluación automática de todas las combinaciones
Salida: 12 csv en outputs/06_postprocesado/
"""

import os
import glob
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.ndimage import label as cc_label
import giotto_tda as gtda

# Configuración matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NNUNET_RAW = Path("data/nnunet_raw")
NNUNET_RESULTS = Path("data/nnunet_results")
IMAGINEM_RAW = Path("data/raw/NEW_LESIONS_IMAGINEM")
MSSEG2_RAW = Path("data/raw/MSSEG-2/LongitudinalMultipleSclerosisLesionSegmentationChallengeMiccai21_v2/training")
OUTPUT_DIR = Path("outputs/06_postprocesado")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar volúmenes ImaginEM para percentiles
df_volumenes = pd.read_csv(IMAGINEM_RAW.parent / "ImaginEM_nLS.csv")
df_volumenes_2 = df_volumenes[df_volumenes['label'] == 2]
percentiles = [1, 3, 5, 10]
volume_mins = {p: np.percentile(df_volumenes_2['volume_mm3'].values, p) for p in percentiles}
volume_min_p3 = volume_mins[3]

print("Configuración postprocesado:")
print(f"  Volume thresholds: {volume_mins}")
print(f"  PH k=3, Vol P3: {volume_min_p3:.2f} mm³")

# Importar funciones evaluación desde Fase 05
exec(open("src/05_metricas/05_evaluacion_completa.py").read())  # voxel/lesion/patient funcs

# ============================================================================
# FUNCIONES POSTPROCESADO VOLUMEN
# ============================================================================

def nifti_get_voxel_volume_mm3(nifti_img):
    """Volumen voxel en mm³ desde header."""
    zooms = nifti_img.header.get_zooms()[:3]
    return float(np.prod(zooms))

def postprocess_vol(mask_img, volume_min_mm3):
    """Filtro volumen solo etiqueta 2."""
    data = mask_img.get_fdata().astype(np.int16)
    voxel_vol = nifti_get_voxel_volume_mm3(mask_img)

    bin_new = (data == 2)
    struct = np.ones((3, 3, 3), dtype=bool)
    cc, n_cc = cc_label(bin_new, structure=struct)

    for i in range(1, n_cc + 1):
        comp = (cc == i)
        vol_mm3 = comp.sum() * voxel_vol
        if vol_mm3 < volume_min_mm3:
            data[comp] = 0

    return nib.Nifti1Image(data.astype(np.int16), mask_img.affine, mask_img.header)

def run_postprocess_vol_only(preds_in, preds_out, volume_min_mm3):
    """Aplica postprocesado volumen a carpeta."""
    os.makedirs(preds_out, exist_ok=True)
    pred_files = sorted(glob.glob(f"{preds_in}/*.nii.gz"))

    for pf in pred_files:
        img = nib.load(pf)
        img_pp = postprocess_vol(img, volume_min_mm3)
        out_path = os.path.join(preds_out, os.path.basename(pf))
        nib.save(img_pp, out_path)

# ============================================================================
# FUNCIONES POSTPROCESADO HOMOLOGÍA PERSISTENTE
# ============================================================================

cp = gtda.homology.CubicalPersistence(homology_dimensions=[0])

def load_flair_followup(imaginem_id):
    """Carga FLAIR followup ImaginEM."""
    path = IMAGINEM_RAW / imaginem_id / f"{imaginem_id}_flair_followup.nii.gz"
    return nib.load(path).get_fdata()

def load_flair_followup_msseg2(msseg2_id):
    """Carga FLAIR followup MSSEG2."""
    path = MSSEG2_RAW / msseg2_id / "flair_time02_on_middle_space.nii.gz"
    return nib.load(path).get_fdata()

def count_bars_h0_flair_bbox(mask_3d, flair_3d):
    """Cuenta barras H0 en bounding box FLAIR."""
    coords = np.where(mask_3d)
    if coords[0].size == 0:
        return 0

    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()

    sub_flair = flair_3d[z0:z1+1, y0:y1+1, x0:x1+1].astype(float)[np.newaxis, ...]
    diags = cp.fit_transform(sub_flair)[0]
    life = diags[diags[:, 2] == 0, 1] - diags[diags[:, 2] == 0, 0]
    life = life[life > 0]
    return int(life.size)

def postprocess_ph_k3(preds_in_dir, preds_out_dir, k=3, dataset="imaginem"):
    """PH k=3: elimina componentes label=2 con <k barras H0."""
    os.makedirs(preds_out_dir, exist_ok=True)
    pred_files = sorted(glob.glob(os.path.join(preds_in_dir, "*.nii.gz")))

    for pf in pred_files:
        name = os.path.basename(pf)
        if dataset == "imaginem":
            imaginem_id = name.replace(".nii.gz", "").replace("ImaginEM_", "")
            flair_fu = load_flair_followup(imaginem_id)
        else:  # msseg2
            msseg2_id = name.replace(".nii.gz", "").replace("MSSEG2_", "")
            flair_fu = load_flair_followup_msseg2(msseg2_id)

        img_pred = nib.load(pf)
        data_pred = img_pred.get_fdata().astype(int)

        pred_bin = (data_pred == 2)
        struct = np.ones((3, 3, 3), dtype=bool)
        cc, num_comp = cc_label(pred_bin, struct)

        for comp_id in range(1, num_comp + 1):
            comp_mask = (cc == comp_id)
            n_bars = count_bars_h0_flair_bbox(comp_mask, flair_fu)
            if n_bars < k:
                data_pred[comp_mask] = 0

        out_path = os.path.join(preds_out_dir, name)
        nib.save(nib.Nifti1Image(data_pred.astype(np.int16), img_pred.affine, img_pred.header), out_path)

# ============================================================================
# PIPELINE POSTPROCESADO
# ============================================================================

def aplicar_postprocesados_completos():
    """Aplica todos los postprocesados y realiza la evaluación de cada uno de ellos con las dos funciones definidas en 05."""
    print("\n" + "="*80)
    print("POSTPROCESADO COMPLETO - MODELO 100 EPOCHS")
    print("="*80)

    base_test = NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/test_holdout_preds_100ep"
    base_msseg2 = NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/msseg2_preds_100ep"
    labels_test = NNUNET_RAW / "Dataset210_ImaginEM_FLAIR_HOLDOUT_TEST/labelsTs"
    labels_msseg2 = NNUNET_RAW / "Dataset302_MSSEG2_FLAIR/labelsTs"

    # 1. FILTRO VOLUMEN (P1,P3,P5,P10)
    print("\n1. FILTRO VOLUMEN P1/P3/P5/P10")
    for p, vmin in volume_mins.items():
        out_test = OUTPUT_DIR / f"test_100ep_VOL_P{p}"
        out_msseg2 = OUTPUT_DIR / f"msseg2_100ep_VOL_P{p}"
        run_postprocess_vol_only(str(base_test), str(out_test), vmin)
        run_postprocess_vol_only(str(base_msseg2), str(out_msseg2), vmin)

    # 2. PH k=3
    print("\n2. HOMOLOGÍA PERSISTENTE k=3")
    ph_test = OUTPUT_DIR / "test_100ep_PH_K3"
    ph_msseg2 = OUTPUT_DIR / "msseg2_100ep_PH_K3"
    postprocess_ph_k3(str(base_test), str(ph_test), dataset="imaginem")
    postprocess_ph_k3(str(base_msseg2), str(ph_msseg2), dataset="msseg2")

    # 3. PH k=3 + Vol P3
    print("\n3. PH k=3 + VOL P3")
    ph_p3_test = OUTPUT_DIR / "test_100ep_PH_K3_VOL_P3"
    ph_p3_msseg2 = OUTPUT_DIR / "msseg2_100ep_PH_K3_VOL_P3"
    run_postprocess_vol_only(str(ph_test), str(ph_p3_test), volume_min_p3)
    run_postprocess_vol_only(str(ph_msseg2), str(ph_p3_msseg2), volume_min_p3)

    # 4. Vol P3 + PH k=3
    print("\n4. VOL P3 + PH k=3")
    vol_p3_test = OUTPUT_DIR / "test_100ep_VOL_P3"
    vol_p3_msseg2 = OUTPUT_DIR / "msseg2_100ep_VOL_P3"
    vol_ph_test = OUTPUT_DIR / "test_100ep_VOL_P3_PH_K3"
    vol_ph_msseg2 = OUTPUT_DIR / "msseg2_100ep_VOL_P3_PH_K3"
    run_postprocess_vol_only(str(base_test), str(vol_p3_test), volume_min_p3)
    run_postprocess_vol_only(str(base_msseg2), str(vol_p3_msseg2), volume_min_p3)
    postprocess_ph_k3(str(vol_p3_test), str(vol_ph_test), dataset="imaginem")
    postprocess_ph_k3(str(vol_p3_msseg2), str(vol_ph_msseg2), dataset="msseg2")

    print(" TODOS POSTPROCESADOS GENERADOS")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(" Fase 06: Postprocesado Completo")
    aplicar_postprocesados_completos()

    print("\n" + " POSTPROCESADO FINALIZADO")
    print(f" En {OUTPUT_DIR}:")
    print("   test_100ep_VOL_P1/P3/P5/P10/")
    print("   msseg2_100ep_VOL_P1/P3/P5/P10/")
    print("   test_100ep_PH_K3/")
    print("   test_100ep_PH_K3_VOL_P3/")
    print("   test_100ep_VOL_P3_PH_K3/")

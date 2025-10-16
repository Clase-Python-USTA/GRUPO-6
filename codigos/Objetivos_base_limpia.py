# -*- coding: utf-8 -*-
# Script listo para VS Code / GitHub Codespaces. Todo se exporta a reports/.

from pathlib import Path
import os, re, json, unicodedata, warnings
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, kruskal

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# -------------------------------------------------------------------
# RUTAS
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "Data"]
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(PROCESADOS_DIR := REPO_ROOT / "data" / "procesados").mkdir(parents=True, exist_ok=True)

def _find_data_file(name: str) -> Path:
    for d in DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No encuentro '{name}' en {DATA_DIRS}")

RAW_FILE     = _find_data_file("base_datos_completa_NNA_TI_anon.xlsx")     # hoja 'BD'
ELEGIDAS_FILE= _find_data_file("DatasetConVariables elegidas.xlsx")        # base depurada

def display(x):  # alias print(tablas) fuera de Colab
    print(x)

print(f"Archivo RAW:      {RAW_FILE}")
print(f"Archivo ELEGIDAS: {ELEGIDAS_FILE}")

# -------------------------------------------------------------------
# Auto-guardar TODAS las figuras (cada plt.show() guarda PNG en reports/)
# + utilidades para exportar tablas a CSV y a un Excel único
# -------------------------------------------------------------------
_fig_id = 1
def _autosave_show(*args, **kwargs):
    global _fig_id
    out = REPORTS_DIR / f"fig_{_fig_id:02d}.png"
    plt.gcf().savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    _fig_id += 1
plt.show = _autosave_show  # monkey-patch

# Excel único de tablas
_xlsx = pd.ExcelWriter(REPORTS_DIR / "tablas.xlsx", engine="openpyxl", mode="w")

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)[:31] or "tabla"

def export_table(df: pd.DataFrame, name: str):
    safe = _safe_name(name)
    df.to_csv(REPORTS_DIR / f"{safe}.csv", index=True, encoding="utf-8")
    df.to_excel(_xlsx, sheet_name=safe)

# -------------------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------------------
SGSSS_ALIASES = [
    "afiliacion_al_sgsss", "afiliacion_sgsss", "afiliación_al_sgsss", "afiliacion_sgss",
    "regimen_de_afiliacion", "regimen_salud", "regimen", "tipo_de_afiliacion", "tipo_afiliacion",
]

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch))

def slug(s: str) -> str:
    s = strip_accents(str(s)).lower().strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def slug_map(cols: List[str]) -> Dict[str, str]:
    m = {}
    for c in cols:
        sc = slug(c)
        i, base = 2, sc
        while sc in m and m[sc] != c:
            sc = f"{base}_{i}"; i += 1
        m[sc] = c
    return m

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.DataFrame({
        "Variable": df.columns,
        "Faltantes": df.isna().sum().values,
        "Porcentaje": (df.isna().mean().values * 100).round(2),
        "Tipo": [str(t) for t in df.dtypes.values],
        "Valores_Unicos": [df[c].nunique(dropna=True) for c in df.columns],
    }).sort_values("Porcentaje", ascending=False))

def is_sentinel_9(val, min_nines: int = 4) -> bool:
    if pd.isna(val): return False
    s = str(val).strip().replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        try:
            v = float(s)
            if v.is_integer():
                s_int = str(int(abs(v)))
                return len(s_int) >= min_nines and set(s_int) == {"9"}
        except Exception:
            return False
    s_digits = re.sub(r"[^0-9]", "", s)
    return len(s_digits) >= min_nines and set(s_digits) == {"9"}

def normalize_localidad(s: str) -> str:
    if pd.isna(s): return np.nan
    s = strip_accents(str(s)).strip()
    s = re.sub(r"[^A-Za-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()

def normalize_sgsss(s: str) -> str:
    if pd.isna(s): return np.nan
    t = strip_accents(str(s)).upper().strip()
    t = re.sub(r"\s+", " ", t)
    if re.search(r"CONTRIBUT", t):   return "CONTRIBUTIVO"
    if re.search(r"SUBSIDI", t):     return "SUBSIDIADO"
    if re.search(r"ESPECIAL|EXCEPC", t): return "ESPECIAL/EXCEPCION"
    if re.search(r"NO\s*AFILI", t):  return "NO AFILIADO"
    if re.search(r"BENEFIC", t):     return "BENEFICIARIO"
    if re.search(r"VINCUL", t):      return "VINCULADO"
    return t

# -------------------------------------------------------------------
# LIMPIEZA (RAW) -> opcional, se continua con ELEGIDAS_FILE para análisis
# -------------------------------------------------------------------
RAW_TARGET_COLUMNS = [
    "AFILIACIÓN AL SGSSS",
    "CATEGORÍAS_DE_LA_DISCAPACIDAD",
    "Localidad_fic",
    "NACIONALIDAD",
    "SEXO",
    "¿EN DONDE REALIZA PRINCIPALMENTE SU TRABAJO?",
    "VÍNCULO CON EL JEFE DE HOGAR",
    "EDAD",
]
CAT_IMPUTE_VALUE = "No especificado"
SHEET_NAME = "BD"

try:
    df_raw = pd.read_excel(RAW_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
except Exception:
    df_raw = pd.read_excel(RAW_FILE, engine="openpyxl")
print(f"Leídas {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas de RAW")

col_slug = slug_map(list(df_raw.columns))
available_slugs = set(col_slug.keys())
targets_resolved: Dict[str, Optional[str]] = {}
for wanted in RAW_TARGET_COLUMNS:
    w_slug = slug(wanted)
    found = col_slug.get(w_slug)
    if not found and w_slug == "afiliacion_al_sgsss":
        for alias in SGSSS_ALIASES:
            if alias in available_slugs:
                found = col_slug[alias]; break
    if not found:
        tokens = [t for t in w_slug.split("_") if t not in {"de","del","la","el","los","las","y","con","su","en","al"}]
        pattern = re.compile("|".join(map(re.escape, tokens)), flags=re.I) if tokens else None
        if pattern:
            for s in available_slugs:
                if pattern.search(s):
                    found = col_slug[s]; break
    targets_resolved[wanted] = found

print("\nMapeo columnas objetivo (RAW):")
for k, v in targets_resolved.items():
    print(f"  - {k} -> {v if v else 'NO ENCONTRADA'}")

present_cols = [v for v in targets_resolved.values() if v]
if present_cols:
    df_clean = df_raw[present_cols].copy()
    for c in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[c] = df_clean[c].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else x)
    df_clean = df_clean.mask(df_clean.applymap(lambda x: is_sentinel_9(x, min_nines=4)), np.nan)
    col_localidad = targets_resolved.get("Localidad_fic")
    if col_localidad and col_localidad in df_clean.columns:
        df_clean[col_localidad] = df_clean[col_localidad].apply(normalize_localidad)
    col_sgsss = targets_resolved.get("AFILIACIÓN AL SGSSS")
    if col_sgsss and col_sgsss in df_clean.columns:
        df_clean.rename(columns={col_sgsss: "AFILIACION_AL_SGSSS"}, inplace=True)
        df_clean["AFILIACION_AL_SGSSS"] = df_clean["AFILIACION_AL_SGSSS"].apply(normalize_sgsss)
    col_edad = targets_resolved.get("EDAD")
    if col_edad and col_edad in df_clean.columns:
        df_clean.rename(columns={col_edad: "EDAD"}, inplace=True)
        df_clean["EDAD"] = pd.to_numeric(df_clean["EDAD"], errors="coerce")
        antes = len(df_clean); df_clean = df_clean[df_clean["EDAD"].notna() & (df_clean["EDAD"] != 0.0)]
        print(f"Eliminadas {antes - len(df_clean):,} filas con EDAD = 0.0")
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in num_cols:
        if df_clean[c].isna().any(): df_clean[c].fillna(df_clean[c].median(), inplace=True)
    for c in cat_cols:
        if df_clean[c].isna().any(): df_clean[c].fillna(CAT_IMPUTE_VALUE, inplace=True)
    print("\n✅ Limpieza RAW terminada")
else:
    print("⚠️  No se hallaron columnas objetivo en RAW; se analiza la base elegida.")

# -------------------------------------------------------------------
# OBJETIVO I  (usa base depurada ELEGIDAS_FILE)
# -------------------------------------------------------------------
df = pd.read_excel(ELEGIDAS_FILE)
display(df.head())

# Grupos etarios
condiciones = [
    (df['EDAD'] >= 0) & (df['EDAD'] <= 5),
    (df['EDAD'] >= 6) & (df['EDAD'] <= 11),
    (df['EDAD'] >= 12) & (df['EDAD'] <= 17)
]
categorias = ['0-5 años (Primera infancia)','6-11 años (Infancia)','12-17 años (Adolescencia)']
df['GRUPO_ETARIO'] = np.select(condiciones, categorias, default='Otro')
print("✓ Variable GRUPO_ETARIO creada")

print("\n" + "="*60); print("DISTRIBUCIÓN DE GRUPOS ETARIOS"); print("="*60 + "\n")
distribucion = df['GRUPO_ETARIO'].value_counts().sort_index()
display(distribucion)
export_table(distribucion.to_frame("Frecuencia"), "distribucion_grupo_etario")

print("\n" + "="*60); print("PORCENTAJES POR GRUPO ETARIO"); print("="*60 + "\n")
porcentajes = (df['GRUPO_ETARIO'].value_counts(normalize=True).sort_index() * 100).round(2)
display(porcentajes)
export_table(porcentajes.to_frame("Porcentaje"), "porcentajes_grupo_etario")

# EDAD: descriptivos y gráficos
sns.set_style("whitegrid"); plt.rcParams['figure.figsize'] = (12, 6); plt.rcParams['font.size'] = 10
print("="*60); print("ANÁLISIS DESCRIPTIVO UNIVARIADO - EDAD"); print("="*60)
print("\n1. ESTADÍSTICOS DESCRIPTIVOS\n")
print(f"Media: {df['EDAD'].mean():.2f}  Mediana: {df['EDAD'].median():.0f}  Moda: {df['EDAD'].mode()[0]:.0f}")
print(f"Desv.Std: {df['EDAD'].std():.2f}  Min: {df['EDAD'].min():.0f}  Max: {df['EDAD'].max():.0f}")

print("\n" + "="*60); print("2. DISTRIBUCIÓN DE FRECUENCIAS POR EDAD"); print("="*60 + "\n")
frecuencias = df['EDAD'].value_counts().sort_index()
frecuencias_rel = df['EDAD'].value_counts(normalize=True).sort_index() * 100
tabla_frecuencias = pd.DataFrame({
    'Frecuencia': frecuencias,
    'Porcentaje': frecuencias_rel.round(2),
    'Frecuencia Acumulada': frecuencias.cumsum(),
    'Porcentaje Acumulado': frecuencias_rel.cumsum().round(2)
})
display(tabla_frecuencias)
export_table(tabla_frecuencias, "tabla_frecuencias_edad")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(df['EDAD'], bins=range(df['EDAD'].min(), df['EDAD'].max() + 2), edgecolor='black', alpha=0.7)
axes[0].axvline(df['EDAD'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["EDAD"].mean():.2f}')
axes[0].axvline(df['EDAD'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["EDAD"].median():.0f}')
axes[0].set_xlabel('Edad (años)'); axes[0].set_ylabel('Frecuencia'); axes[0].set_title('Distribución de Edad'); axes[0].legend()
edad_counts = df['EDAD'].value_counts().sort_index()
axes[1].bar(edad_counts.index, edad_counts.values, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Edad (años)'); axes[1].set_ylabel('Número de casos'); axes[1].set_title('Frecuencia por Edad'); axes[1].set_xticks(range(df['EDAD'].min(), df['EDAD'].max() + 1))
plt.tight_layout(); plt.show()

print("\n" + "="*60); print("3. ANÁLISIS DE EDADES CRÍTICAS"); print("="*60 + "\n")
menores_5 = df[df['EDAD'] < 5].shape[0]
entre_5_14 = df[(df['EDAD'] >= 5) & (df['EDAD'] < 15)].shape[0]
adolescentes = df[(df['EDAD'] >= 15) & (df['EDAD'] <= 17)].shape[0]
print(f"0-4: {menores_5} ({menores_5/len(df)*100:.2f}%)  5-14: {entre_5_14} ({entre_5_14/len(df)*100:.2f}%)  15-17: {adolescentes} ({adolescentes/len(df)*100:.2f}%)")

plt.figure(figsize=(10, 6))
plt.boxplot(df['EDAD'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='red', linewidth=2))
plt.xlabel('Edad (años)'); plt.title('Diagrama de Caja - EDAD'); plt.grid(axis='x', alpha=0.3); plt.yticks([])
plt.tight_layout(); plt.show()

# SEXO
sns.set_style("whitegrid"); plt.rcParams['figure.figsize'] = (12, 6); plt.rcParams['font.size'] = 10
print("="*60); print("ANÁLISIS DESCRIPTIVO UNIVARIADO - SEXO"); print("="*60)
frecuencias = df['SEXO'].value_counts(); frecuencias_rel = df['SEXO'].value_counts(normalize=True) * 100
tabla_sexo = pd.DataFrame({'Frecuencia': frecuencias,'Porcentaje': frecuencias_rel.round(2)})
display(tabla_sexo); export_table(tabla_sexo, "tabla_sexo")
print(f"\nTotal de casos: {len(df)}")
hombres = df[df['SEXO'] == '1- Hombre'].shape[0]; mujeres = df[df['SEXO'] == '2- Mujer'].shape[0]; intersexual = df[df['SEXO'] == '3- Intersexual'].shape[0]
if mujeres > 0: razon_masculinidad = (hombres / mujeres) * 100
else: razon_masculinidad = np.nan

fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(frecuencias.index, frecuencias.values, edgecolor='black', alpha=0.8, width=0.6)
for i, (barra, valor) in enumerate(zip(barras, frecuencias.values)):
    height = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2., height, f'{valor}\n({frecuencias_rel.values[i]:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Número de casos'); ax.set_xlabel('Sexo'); ax.set_title('Distribución de Casos por Sexo'); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, max(frecuencias.values) * 1.15)
plt.tight_layout(); plt.show()

print("\n" + "="*60); print("3. INTERPRETACIÓN"); print("="*60 + "\n")
if pd.notna(razon_masculinidad):
    if abs(razon_masculinidad - 100) < 10: print("Distribución por sexo equilibrada.")
    elif razon_masculinidad > 110: print("Predominio masculino.")
    else: print("Predominio femenino.")

# NACIONALIDAD
print("="*60); print("ANÁLISIS DESCRIPTIVO UNIVARIADO - NACIONALIDAD"); print("="*60)
frecuencias = df['NACIONALIDAD'].value_counts(); frecuencias_rel = df['NACIONALIDAD'].value_counts(normalize=True) * 100
tabla_nacionalidad = pd.DataFrame({
    'Frecuencia': frecuencias,
    'Porcentaje': frecuencias_rel.round(2),
    'Frecuencia Acumulada': frecuencias.cumsum(),
    'Porcentaje Acumulado': frecuencias_rel.cumsum().round(2)
})
display(tabla_nacionalidad); export_table(tabla_nacionalidad, "tabla_nacionalidad")
print(f"\nTotal de casos: {len(df)}"); print(f"Número de nacionalidades: {df['NACIONALIDAD'].nunique()}")

print("\n" + "="*60); print("2. COLOMBIA VS MIGRANTES"); print("="*60 + "\n")
colombianos = df[df['NACIONALIDAD'] == 'Colombia'].shape[0]; migrantes = df[df['NACIONALIDAD'] != 'Colombia'].shape[0]
pct_col = (colombianos / len(df)) * 100; pct_mig = (migrantes / len(df)) * 100
print(f"Colombianos: {colombianos} ({pct_col:.2f}%)  Migrantes: {migrantes} ({pct_mig:.2f}%)")

df_migrantes = df[df['NACIONALIDAD'] != 'Colombia']
if len(df_migrantes) > 0:
    paises_migrantes = df_migrantes['NACIONALIDAD'].value_counts()
    paises_migrantes_pct = df_migrantes['NACIONALIDAD'].value_counts(normalize=True) * 100
    tabla_migrantes = pd.DataFrame({
        'Frecuencia': paises_migrantes,
        'Porc. dentro migrantes': paises_migrantes_pct.round(2),
        'Porc. del total': (paises_migrantes / len(df) * 100).round(2)
    })
    display(tabla_migrantes); export_table(tabla_migrantes, "tabla_paises_migrantes")

# -------------------------------------------------------------------
# CRUCES RESUMIDOS: Pirámide EDAD x SEXO
# -------------------------------------------------------------------
print("="*70); print("ANÁLISIS BIVARIADO RESUMIDO"); print("="*70)
print("\nPirámide poblacional (Edad x Sexo)")
edades = sorted(df['EDAD'].unique()); hombres_por_edad=[]; mujeres_por_edad=[]
for edad in edades:
    hombres_por_edad.append(-df[(df['EDAD']==edad) & (df['SEXO']=='1- Hombre')].shape[0])
    mujeres_por_edad.append( df[(df['EDAD']==edad) & (df['SEXO']=='2- Mujer')].shape[0])
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(edades, hombres_por_edad, height=0.8, alpha=0.8, edgecolor='black', label='Hombres')
ax.barh(edades, mujeres_por_edad, height=0.8, alpha=0.8, edgecolor='black', label='Mujeres')
ax.set_xlabel('Número de casos'); ax.set_ylabel('Edad (años)'); ax.set_title('Pirámide Poblacional'); ax.axvline(x=0, color='black', linewidth=1.5); ax.legend(loc='upper right')
max_val = max(max(abs(min(hombres_por_edad)), max(mujeres_por_edad)), 1)
ticks = range(0, int(max_val) + 5, max(1, int(max_val/5)))
xticks_vals  = ([-t for t in ticks][::-1] + list(ticks))
xtick_labels = [str(abs(t)) for t in xticks_vals]
ax.set_xticks(xticks_vals); ax.set_xticklabels(xtick_labels)
plt.tight_layout(); plt.show()

# -------------------------------------------------------------------
# OBJETIVO II  (familia y acompañamiento)
# -------------------------------------------------------------------
col_lugar = '¿EN DONDE REALIZA PRINCIPALMENTE SU TRABAJO?'
col_sgsss = 'AFILIACION_AL_SGSSS'
col_vinc  = 'VÍNCULO CON EL JEFE DE HOGAR'

for c in [col_lugar, col_sgsss, col_vinc]:
    print(c, '→ n=', df[c].notna().sum(), 'categorías=', df[c].nunique(dropna=True))

vc = df[col_lugar].value_counts().sort_values(ascending=False)
labels = vc.index.astype(str); vals = (100*vc/vc.sum()).values
plt.figure(figsize=(8,4))
bars = plt.bar(labels, vals)
plt.xticks(rotation=45, ha='right'); plt.ylabel('%'); plt.title('¿Dónde se presenta el trabajo infantil?')
for b, v in zip(bars, vals):
    plt.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
plt.ylim(0, max(vals)*1.15); plt.tight_layout(); plt.show()

def barras_top3(var, var_lugar, titulo=None):
    ct = pd.crosstab(df[var], df[var_lugar])
    cols_ord = ct.sum().sort_values(ascending=False).index.tolist()
    cols_ord = [c for c in cols_ord if str(c).strip().lower()!='no especificado']
    top3 = cols_ord[:3]; otros = [c for c in ct.columns if c not in top3]
    ct2 = pd.DataFrame(index=ct.index)
    for c in top3: ct2[c] = ct[c]
    ct2['Otros'] = ct[otros].sum(axis=1)
    pct = (ct2.div(ct2.sum(axis=1), axis=0)*100).fillna(0)
    pct = pct.sort_values(next((c for c in pct.columns if 'uti' in c.lower()), pct.columns[0]), ascending=False)
    x = np.arange(len(pct)); w = 0.22
    plt.figure(figsize=(9,4))
    for i,col in enumerate(pct.columns):
        plt.bar(x + (i-1.5)*w, pct[col].values, width=w, label=col)
        for j,val in enumerate(pct[col].values):
            if val>=5: plt.text(x[j] + (i-1.5)*w, val+1, f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    plt.xticks(x, pct.index.astype(str), rotation=45, ha='right'); plt.ylabel('%'); plt.ylim(0, 100)
    plt.title(titulo or f'{var}: composición por lugar (Top-3 + Otros)'); plt.legend(ncol=4, fontsize=9); plt.tight_layout(); plt.show()
    return pct.reset_index()

pct_sgsss = barras_top3(col_sgsss, col_lugar, 'AFILIACION_AL_SGSSS × lugar de trabajo')
export_table(pct_sgsss, "pct_sgsss_top3_lugares")

# Vínculo × lugar (apiladas 100%)
var = col_vinc
ct = pd.crosstab(df[var], df[col_lugar])
cols_ord = ct.sum().sort_values(ascending=False).index.tolist()
cols_ord = [c for c in cols_ord if str(c).strip().lower()!='no especificado']
top3 = cols_ord[:3]; otros = [c for c in ct.columns if c not in top3]
ct2 = pd.DataFrame(index=ct.index)
for c in top3: ct2[c] = ct[c]
ct2['Otros'] = ct[otros].sum(axis=1)
pct = (ct2.div(ct2.sum(axis=1), axis=0)*100).fillna(0)
col_uti = next((c for c in pct.columns if 'uti' in c.lower()), pct.columns[0])
pct = pct.sort_values(col_uti, ascending=False)
export_table(pct.reset_index(), "pct_vinculo_x_lugar_top3")
plt.figure(figsize=(9, 0.45*len(pct)+1))
left = np.zeros(len(pct))
for col in pct.columns:
    plt.barh(pct.index.astype(str), pct[col].values, left=left, label=col)
    for i, v in enumerate(pct[col].values):
        if v >= 8: plt.text(left[i]+v/2, i, f'{v:.0f}%', va='center', ha='center', fontsize=8)
    left += pct[col].values
plt.xlabel('%'); plt.title('VÍNCULO × lugar de trabajo — composición (Top-3 + Otros)'); plt.legend(ncol=4, fontsize=9); plt.xlim(0,100); plt.tight_layout(); plt.show()

def top10_combos(var, var_lugar):
    ct = pd.crosstab(df[var], df[var_lugar])
    tall = ct.stack().reset_index(); tall.columns = [var, 'lugar', 'n']
    denom = ct.sum(axis=1).rename('n_categoria').reset_index()
    out = tall.merge(denom, on=var, how='left')
    out['% dentro de la categoría'] = 100*out['n']/out['n_categoria']
    out['% del total'] = 100*out['n']/len(df)
    out = out.sort_values('n', ascending=False).head(10)
    return out

top_sgsss = top10_combos(col_sgsss, col_lugar)
top_vinc  = top10_combos(col_vinc,  col_lugar)
print("\nTOP SGSSS x Lugar (Top10)"); display(top_sgsss)
print("\nTOP VÍNCULO x Lugar (Top10)"); display(top_vinc)
export_table(top_sgsss, "top10_sgsss_x_lugar")
export_table(top_vinc,  "top10_vinculo_x_lugar")

# -------------------------------------------------------------------
# OBJETIVO III  (localidades)
# -------------------------------------------------------------------
df_loc = pd.read_excel(ELEGIDAS_FILE)
df_loc.columns = df_loc.columns.str.strip().str.upper()
col_loc = "LOCALIDAD_FIC"; col_edad = "EDAD"; col_sexo = "SEXO"; col_nac = "NACIONALIDAD"
df_loc[col_edad] = pd.to_numeric(df_loc[col_edad], errors='coerce')
ninos = df_loc[df_loc[col_edad] < 18].copy()
ninos[col_loc] = ninos[col_loc].astype(str).str.strip().str.upper()

print(f"\n✅ Total de registros: {len(df_loc)}  |  Niños (<18): {len(ninos)}\n")

conteo_localidad = ninos[col_loc].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10,6)); sns.barplot(x=conteo_localidad.values, y=conteo_localidad.index)
plt.title("Número de niños por localidad"); plt.xlabel("Cantidad de niños"); plt.ylabel("Localidad"); plt.tight_layout(); plt.show()
print("\nDistribución de niños por localidad:"); display(conteo_localidad)
export_table(conteo_localidad.to_frame("Cantidad"), "ninos_por_localidad")

ninos[col_sexo] = ninos[col_sexo].astype(str).str.strip().str.upper()
tabla_sexo_loc = pd.crosstab(ninos[col_loc], ninos[col_sexo], normalize="index") * 100
plt.figure(figsize=(10,6)); sns.heatmap(tabla_sexo_loc, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("% por sexo dentro de cada localidad"); plt.xlabel("Sexo"); plt.ylabel("Localidad"); plt.tight_layout(); plt.show()
print("\nPorcentaje por sexo dentro de cada localidad:"); display(tabla_sexo_loc.round(1))
export_table(tabla_sexo_loc.round(1), "sexo_por_localidad")

ninos[col_nac] = ninos[col_nac].astype(str).str.strip().str.upper()
ninos["NAC_SIMPLE"] = ninos[col_nac].apply(lambda x: "COLOMBIANA" if "COL" in x else "EXTRANJERA")
tabla_nac = pd.crosstab(ninos[col_loc], ninos["NAC_SIMPLE"], normalize="index") * 100
plt.figure(figsize=(10,6)); sns.heatmap(tabla_nac, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("% de nacionalidad por localidad"); plt.xlabel("Nacionalidad"); plt.ylabel("Localidad"); plt.tight_layout(); plt.show()
print("\nPorcentaje de nacionalidad por localidad:"); display(tabla_nac.round(1))
export_table(tabla_nac.round(1), "nacionalidad_por_localidad")

# -------------------------------------------------------------------
# PRUEBAS ESTADÍSTICAS + exportes
# -------------------------------------------------------------------
def cramers_v(chi2, n, r, c):
    k = min(r-1, c-1)
    return np.sqrt(chi2 / (n * k)) if k > 0 else np.nan

tests: Dict[str, Any] = {}

# Chi2 Localidad × Sexo (usar tabla de % por localidad)
ct1 = tabla_sexo_loc
chi2_1, p1, dof1, exp1 = chi2_contingency(ct1)
cram1 = cramers_v(chi2_1, ct1.values.sum(), *ct1.shape)
tests["Localidad×Sexo"] = {"chi2": float(chi2_1), "p": float(p1), "cramers_v": float(cram1)}

# Chi2 Localidad × Nacionalidad
ct2 = tabla_nac
chi2_2, p2, dof2, exp2 = chi2_contingency(ct2)
cram2 = cramers_v(chi2_2, ct2.values.sum(), *ct2.shape)
tests["Localidad×Nacionalidad"] = {"chi2": float(chi2_2), "p": float(p2), "cramers_v": float(cram2)}

# Kruskal–Wallis Edad × Localidad
grupos = [g[col_edad].dropna().values for _, g in ninos.groupby(col_loc) if len(g[col_edad].dropna()) > 1]
if len(grupos) >= 2:
    H, p3 = kruskal(*grupos)
    N = int(sum(len(g) for g in grupos))
    k = int(len(grupos))
    eps2 = float((H - k + 1) / (N - k))
    tests["Edad×Localidad"] = {"H": float(H), "p": float(p3), "epsilon²": eps2}
else:
    tests["Edad×Localidad"] = {"error": "No hay suficientes localidades con más de un dato."}

print("\n=== RESULTADOS ESTADÍSTICOS ===")
for t, vals in tests.items():
    print(f"\n▶ {t}")
    for k, v in vals.items():
        print(f"   {k}: {v}")

with open(REPORTS_DIR / "tests_results.json", "w", encoding="utf-8") as f:
    json.dump(tests, f, indent=2, ensure_ascii=False)

with open(REPORTS_DIR / "tests_summary.txt", "w", encoding="utf-8") as f:
    for t, vals in tests.items():
        f.write(f"\n▶ {t}\n")
        for k, v in vals.items():
            f.write(f"   {k}: {v}\n")

# -------------------------------------------------------------------
# CIERRE
# -------------------------------------------------------------------
_xlsx.close()
print(f"\n✅ Todo exportado en: {REPORTS_DIR}")

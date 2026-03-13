# Protein Mutation Stability Prediction (SKEMPI v2)

This research project aims to predict the change in binding free energy ($\Delta\Delta G$) upon single-point mutations in protein-protein complexes using machine learning. We explore a dual-resolution approach that merges biophysical feature engineering (Context Window Method) with high-dimensional transformer self-attention (ESM-2) to capture both local chemical disruptions and macro-structural folding constraints.

## 🚀 Overview

The project uses the **SKEMPI v2** dataset, specifically focusing on single-point mutations. We've implemented a robust pipeline that extracts biophysical properties, incorporates neighboring residues' hydrophobicity, and leverages a 150-million parameter Protein Language Model. The final architecture relies on a **Combined Multi-Layer Perceptron (MLP)** evaluated through rigorous Grouped 5-Fold Cross-Validation.

## 🛠️ Methodology

### 1. Data Preparation
- **Dataset**: SKEMPI v2 (raw files under `datasets/`, cleaned for single mutations).
- **Features**: 
  - `delta_vol`: Change in residue volume.
  - `delta_hydro`: Change in hydrophobicity.
  - `loc`: Encoded mutation location (Surface, Core, Support).
  - *Note on Temperature*: While necessary for thermodynamic calculations, temperature was excluded as a predictive feature. Since most assays are performed near 298K, this variable lacked meaningful variance and acted as artificial noise.

### 2. Data Inclusion & Exclusion Criteria
To ensure high-quality training and relevance to single-point mutation studies, we applied the following filters to the raw SKEMPI database:

#### **What we are using (Included):**
- **Single-Point Mutations Only**: We specifically isolate entries where only one amino acid is changed.
- **Complete Biophysical Data**: Only mutations with valid WT and Mutant Affinity values ($\Delta G$) and temperature data are included to calculate the target $\Delta\Delta G$.
- **Protein-Protein Complexes**: We focus on interactions between distinct protein chains as defined in the PDB.
- **Validated Sequences**: Only mutations for which we could retrieve a matching FASTA sequence for the specific chain are processed.

#### **What we have excluded:**
- **Multiple Mutations**: Entries with multiple simultaneous mutations (e.g., `A23G, L45V`) are filtered out.
- **Missing Affinity/Temperature**: Rows with `NaN` or invalid formatted entries are dropped.
- **Unreachable Sequences**: PDB IDs that do not have corresponding FASTA files.
- **Non-Standard Residues**: Mutations involving non-canonical amino acids.

### 3. Context Window Method
To capture the local environment, we implemented a sequence-based windowing approach:
- Extract a **Window of 3 residues** on either side ($L_1, L_2, L_3$ and $R_1, R_2, R_3$).
- Hydrophobicity of these neighboring residues acts as 6 additional features.

### 4. ESM-2 Transformer Embeddings
For advanced macro-structural context, we use the **ESM-2 (Evolutionary Scale Modeling)** language model:
- **Model**: `esm2_t30_150M_UR50D` (Hugging Face, 150M parameters).
- **Extraction**: We extract the 640-dimensional hidden state vector exactly at the mutation site for both WT and Mutant sequences.
- **Delta Vector**: The element-wise difference between mutant and WT embeddings is used to represent the "functional shift."

### 5. The Combined Architecture & Rigorous Evaluation
Our final input vector consists of **649 features** (640 ESM + 6 Window + 3 Base). To prevent data leakage, we evaluate this using a **Grouped Split (by PDB ID)**:
- **1-Fold Peak Split**: An initial 80/20 grouped split to establish the model's performance ceiling.
- **Grouped 5-Fold CV**: Evaluates the model strictly on its ability to generalize thermodynamic rules to entirely unseen structural families.

## 📊 Results

The following results compare baseline models against the **Combined MLP** (512, 256, 64 architecture).

### 1-Fold Peak Performance
On a single optimized split, the architecture demonstrates its high-end predictive capacity:
- **Combined 150M**: **0.651** MAE (kcal/mol) | **0.664** Pearson $r$

### Grouped 5-Fold Cross-Validation (Generalized Performance)
To establish realistic real-world deployment metrics across unseen protein complexes, we aggregated the performance over 5 distinct grouped folds:

| Model Architecture | MAE (kcal/mol) | Pearson $r$ |
| :--- | :---: | :---: |
| **Mutation Only** | 0.912 | 0.145 |
| **Window Only** | 0.935 | 0.000 |
| **ESM-2 Only (150M)** | 0.935 | 0.294 |
| **Random Forest** | **0.900** | 0.284 |
| **Combined MLP** | 0.920 | **0.359** |

### Key Findings:
- The **Combined MLP** successfully leverages the 640-dimensional self-attention space alongside explicit local window properties, achieving the best generalized correlation (Pearson $r$ = 0.359).
- The juxtaposition of the 1-fold peak (0.664) versus the 5-fold average (0.359) underscores the immense structural diversity of the PDB, proving that generalized stability prediction requires rigorous cross-validation to avoid overstating performance.
- While the Random Forest model slightly edges out the MLP in absolute error bounding (0.900 MAE), the neural network is significantly superior at ranking structural stability variance.

## 🏃 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn biopython scipy torch transformers matplotlib
   ```

2. **Prepare Data**:
   ```bash
   python3 data_prep/skempi.py
   python3 data_prep/download_fasta.py
   ```
   This produces `datasets/skempi_cleaned_single_muts.csv` and populates `fasta_files/`.

3. **Generate Features**:
   ```bash
   python3 data_prep/context_window_method.py
   python3 data_prep/context_abs.py       # Optional: absolute + delta window features
   python3 data_prep/ml_model_dataprep.py # Baseline ML features
   python3 data_prep/feature.py           # Extracts 150M ESM-2 embeddings
   ```

4. **Train and Evaluate**:
   ```bash
   python3 models/train_mlp_combined.py
   python3 models/train_mlp_model.py
   python3 models/train_baseline_models.py
   python3 models/train_baseline_temperature.py
   python3 models/train_window_model.py
   python3 models/advance.py
   ```

5. **Generate Visualizations**:
   ```bash
   python3 data_visual/compare.py  # Generates plots/graph.png
   python3 data_visual/graph_5.py  # Generates plots/5fold_graph.png
   ```

**Note**: All scripts resolve paths relative to the repository root and read/write data in `datasets/`, `fasta_files/`, and `plots/`.

## 📂 Project Structure
- `data_prep/`: Data cleaning, FASTA download, and feature engineering scripts.
- `data_prep/skempi.py`: Cleans SKEMPI export and computes ΔΔG.
- `data_prep/download_fasta.py`: Downloads PDB FASTA sequences into `fasta_files/`.
- `data_prep/context_window_method.py`: ±3 window delta features.
- `data_prep/context_abs.py`: Window features with absolute + delta values.
- `data_prep/ml_model_dataprep.py`: Baseline ML feature table.
- `data_prep/feature.py`: ESM-2 (150M) embeddings at mutation site.
- `models/`: Training and analysis scripts.
- `models/train_mlp_combined.py`: Combined MLP with grouped 5-fold CV.
- `models/train_mlp_model.py`: Single split combined MLP.
- `models/train_baseline_models.py`: Baseline linear + RF models.
- `models/train_baseline_temperature.py`: Baseline models with temperature feature.
- `models/train_window_model.py`: Window-only model with absolute features.
- `models/advance.py`: Advanced analysis (feature importance, PCA).
- `data_visual/compare.py` and `data_visual/graph_5.py`: Plot generators.
- `datasets/`: Raw and generated CSV/XLSX files (e.g., `skempi.csv`, `skempi_cleaned_single_muts.csv`, `skempi_esm2_features.csv`, `skempi_window_features.csv`, `skempi_window_features_abs.csv`, `skempi_ml_baseline_features.csv`).
- `fasta_files/`: PDB sequences in FASTA format.
- `plots/`: Generated figures (e.g., `graph.png`, `5fold_graph.png`, `scatter_grouped_combined.png`).

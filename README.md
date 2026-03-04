# Protein Mutation Stability Prediction (SKEMPI v2.0)

This research project aims to predict the change in binding free energy ($\Delta\Delta G$) upon single-point mutations in protein-protein complexes using machine learning. We explore biophysical feature engineering and a **Context Window Method** to capture local sequence environment.

## 🚀 Overview

The project uses the **SKEMPI v2.0** dataset, specifically focusing on single-point mutations. We've implemented a pipeline that extracts biophysical properties (hydrophobicity, volume change) and incorporates neighboring residues' properties to improve prediction accuracy.

## 🛠️ Methodology

### 1. Data Preparation
- **Dataset**: SKEMPI v2.1 (cleaned for single mutations).
- **Features**: 
  - `delta_vol`: Change in residue volume.
  - `delta_hydro`: Change in hydrophobicity.
  - `loc`: Encoded mutation location (Surface, Core, Support).
  - `delta_charge`: Change in net charge (baseline only).

### 2. Data Inclusion & Exclusion Criteria
To ensure high-quality training and relevance to single-point mutation studies, we applied the following filters to the raw SKEMPI database:

#### **What we are using (Included):**
- **Single-Point Mutations Only**: We specifically isolate entries where only one amino acid is changed.
- **Complete Biophysical Data**: Only mutations with valid WT and Mutant Affinity values ($\Delta G$) and temperature data are included to calculate the target $\Delta\Delta G$.
- **Protein-Protein Complexes**: We focus on interactions between distinct protein chains as defined in the PDB.
- **Validated Sequences**: Only mutations for which we could retrieve a matching FASTA sequence for the specific chain are processed for the window model.

#### **What we have excluded:**
- **Multiple Mutations**: Entries with multiple simultaneous mutations (e.g., `A23G, L45V`) are filtered out to avoid confounding effects.
- **Missing Affinity/Temperature**: Rows with `NaN` or invalid formatted entries in the affinity or temperature columns are dropped.
- **Unreachable Sequences**: PDB IDs that do not have corresponding FASTA files or where the mutation position falls outside the retrieved sequence range.
- **Non-Standard Residues**: Any mutations involving non-canonical amino acids that lack standard hydrophobicity or volume metrics.

### 3. Context Window Method
To capture the local environment, we implemented a sequence-based windowing approach:
- For each mutation, we extract a **Window of 3 residues** on either side ($L_1, L_2, L_3$ and $R_1, R_2, R_3$).
- Hydrophobicity of these neighboring residues is used as additional features.
- This allows the model to "see" the chemical environment surrounding the mutation site.

### 4. ESM-2 Transformer Embeddings
For advanced sequence context, we use the **ESM-2 (Evolutionary Scale Modeling)** protein language model:
- **Model**: `esm2_t6_8M_UR50D` (Hugging Face).
- **Extraction**: We extract the 320-dimensional hidden state vector specifically at the mutation site for both WT and Mutant sequences.
- **Delta Vector**: The difference between mutant and WT embeddings is used as a feature, representing the "functional shift" caused by the mutation.

### 5. Rigorous Evaluation
We use a **Grouped Split (by PDB ID)** to prevent data leakage. This ensures the model is tested on entirely different protein complexes than those used during training.

## 📊 Results So Far

The following results were achieved using a **Random Forest Regressor** with a Grouped 80/20 train-test split.

| Model | MSE | MAE (kcal/mol) | Pearson $r$ |
| :--- | :---: | :---: | :---: |
| **Baseline (No Context)** | 2.302 | 1.016 | 0.365 |
| **Windowed Model (RF)** | **1.018** | **0.717** | **0.285** |
| **MLP (Window Features)** | 1.180 | 0.818 | 0.127 |
| **ESM-2 Embeddings** | *In Progress* | *In Progress* | *In Progress* |

### Key Findings:
- The **Context Window Method** with Random Forest remains our best-performing approach, achieving an MAE of **0.717 kcal/mol**.
- The **MLP (Neural Network)** showed competitive results (MAE: 0.818) but struggled with correlation compared to the Random Forest, suggesting that the non-linear decision boundaries of RF are better suited for this limited feature set.
- Pearson correlation remains a technical challenge across all models, likely due to the complexity of protein interactions not captured by sequence-only features.

## 🏃 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn biopython scipy torch transformers
   ```
2. **Prepare Data**:
   Ensure `skempi_cleaned_single_muts.csv` and the `fasta_files/` directory are present.
3. **Generate Features**:
   ```bash
   # Biophysical Window Features
   python3 context_window_method.py
   
   # ESM-2 Transformer Features
   python3 feature.py
   ```
4. **Train and Evaluate**:
   ```bash
   # Random Forest Window Model
   python3 train_window_model.py

   # MLP Neural Network Model
   python3 train_mlp_model.py
   ```

## 📂 Project Structure
- `context_window_method.py`: Feature extraction using the sequence window.
- `feature.py`: Script using ESM-2 to extract transformer embeddings.
- `train_window_model.py`: Random Forest training with context features.
- `train_mlp_model.py`: MLP Neural Network training.
- `train_baseline_models.py`: Baseline comparisons (Linear Regression vs. Random Forest).
- `fasta_files/`: PDB sequences in FASTA format.
- `skempi_esm2_features.csv`: High-dimensional embedding dataset.

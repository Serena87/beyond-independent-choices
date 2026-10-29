# Beyond Independent Choices

Source code for the project:
** Beyond Independent Choices: Modeling Bracketing-Like Shopping Episodes in Fashion Recommendation**

# Source code and run instructions

## Files

- `01_data_preparation_eda.ipynb`
  - filters the H&M transaction data to online purchases;
  - constructs customer-day shopping episodes;
  - creates the article metadata table;
  - reproduces the EDA statistics, concentration measures, episode-type frequencies and strong-grouping percentages;
  - saves the processed Parquet files used by the experiment notebook.

- `02_recommender_experiment.ipynb`
  - constructs the chronological warm-up/train/validation/test split;
  - creates fixed warm-up historical features;
  - trains the final 400,000-episode Logistic Regression models;
  - evaluates the 20,000-episode validation and final LR test sets;
  - runs the hard-negative robustness check;
  - runs the post-hoc episode-type, episode-size, concentration, Spearman/FDR and controlled-regression analyses;
  - trains and evaluates the LightGBM robustness models on the fresh 38,085-episode test holdout;
  - calculates feature importance;
  - verifies the final ranking metrics with LensKit.

- `requirements.txt`
  - Python package versions used in the final project environment.

The original working notebooks contained exploratory checks, duplicated cells, package-install cells and intermediate development experiments. Those have been removed from the cleaned versions. The preliminary 200,000-episode LR development run is also omitted because the final reported models use 400,000 training episodes and the pilot does not feed any final result table.

## 1. Folder structure

Use the following project structure:

```
project/
├─ data/
│  ├─ raw/
│  │  ├─ transactions.csv
│  │  └─ articles.csv
│  └─ processed/
├─ results/
├─ notebooks/
│  ├─ 01_data_preparation_eda.ipynb
│  └─ 02_recommender_experiment.ipynb
└─ requirements.txt
```

The Kaggle H&M file is normally named `transactions_train.csv`. The first notebook accepts either `transactions.csv` or `transactions_train.csv`.

`customers.csv` is not required because customer metadata is not used in the final analysis.

## 2. Install the environment

The project was run with Python 3.13.14. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

If Jupyter is not already installed:

```bash
pip install jupyterlab
```

## 3. Download the H&M data

Download the H&M Personalized Fashion Recommendations dataset from Kaggle and place:

- `transactions_train.csv` (or rename it to `transactions.csv`)
- `articles.csv`

in `data/raw/`.

The notebooks do not use the product images or customer metadata.

## 4. Run the notebooks in order

### First: data preparation and EDA

Open and run all cells in:

```
01_data_preparation_eda.ipynb
```

It creates:

```
data/processed/transactions_online.parquet
data/processed/multi-item_shopping_episodes.parquet
data/processed/article_features.parquet
data/processed/group_stats.parquet
```

### Second: recommender experiment

Then restart the kernel and run all cells in:

```
02_recommender_experiment.ipynb
```

The notebook uses `RANDOM_SEED = 42` throughout.

It recreates the final:
- 400,000 training episodes;
- 20,000 validation episodes;
- 20,000 Logistic Regression test episodes;
- remaining 38,085 eligible test episodes as the fresh LightGBM holdout;
- 10 training negatives per episode;
- 99 validation/test negatives per episode;
- customer-clustered bootstrap CIs with 1,000 resamples.

Final tables and score files are saved under:

```
project/results/
```

## 5. Important reproducibility notes

### Fixed historical cutoff

Historical customer and item predictors are calculated only from the warm-up period and remain fixed during training, validation and testing. The code intentionally does not use an expanding history.

### Candidate evaluation

Evaluation is sampled-candidate evaluation, not full-catalog ranking. Validation and test episodes contain one observed positive target and 99 sampled unobserved alternatives.

### Target preparation

One eligible unique article ID is sampled as the target per episode. All occurrences of that target article are removed from the visible context to prevent target leakage.

### Resource use

The final experiment is memory-intensive because the 400,000 training episodes generate 4.4 million candidate rows and validation/test feature tables each contain millions of rows. Running the notebooks from a fresh kernel and closing other memory-intensive applications is recommended.

## 6. Expected final results

Small numerical differences can occur only if package versions, input files, or random seeds differ. With the versions and seed above, the main final results should reproduce the report values, including approximately:

- Logistic Regression NDCG@10: 0.399 independent vs 0.582 episode-context;
- LightGBM NDCG@10: 0.468 independent vs 0.644 episode-context;
- episode-context LightGBM features: about 58.6% of total gain;
- LensKit verification discrepancies approximately zero.





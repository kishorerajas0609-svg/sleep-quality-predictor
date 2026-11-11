# Sleep Quality Predictor (SleepIQ)

SleepIQ is a Streamlit web application that estimates sleep quality as **Good**, **Average**, or **Poor** from sleep schedule, lifestyle, and well-being inputs. It combines a scikit-learn classification model with rule-based sleep-habit recommendations and a summary chart of the dataset's sleep-quality labels.

> **Scope:** This project provides an informational estimate, not a medical assessment or diagnosis.

## Problem statement

Sleep quality is influenced by several connected factors, including sleep duration and timing, stress, caffeine, exercise, screen use, mood, and interruptions. This project turns those inputs into an accessible prediction and practical, rules-based suggestions, with the goal of helping users reflect on habits that may affect their sleep.

## Objectives

- Collect a sleep and lifestyle profile through a simple web interface.
- Apply consistent feature engineering and scaling to inputs before classification.
- Compare multiple classification algorithms and retain the best model by macro F1-score.
- Show a predicted quality label, model confidence when available, and relevant recommendations.
- Present the Good/Average/Poor label distribution for the data loaded by the application.

## Implemented features

- **Interactive Streamlit form:** sleep duration, bedtime, wake-up time, caffeine intake, exercise duration, pre-bed screen time, stress level, mood, and sleep interruptions.
- **Three-class prediction:** produces a Good, Average, or Poor sleep-quality label using the saved model bundle.
- **Confidence display:** shows the predicted class probability when the selected estimator supports `predict_proba` and the probability can be determined.
- **Personalized rule-based guidance:** `recommendations.py` checks the submitted values and provides reasons and tips for factors such as late bedtime, short or long sleep duration, screen use, stress, exercise, caffeine, interruptions, and mood.
- **Dataset summary:** displays the record count and counts/percentages for each sleep-quality class, with an interactive Plotly donut chart.
- **Model comparison and selection:** `train_model.py` trains Logistic Regression, Decision Tree, Random Forest, and RBF-kernel Support Vector Machine classifiers; it reports accuracy, macro precision, macro recall, macro F1, and confusion matrices, then saves the top macro-F1 model and associated preprocessing information.
- **Preprocessing and feature engineering:** `preprocess.py` encodes categorical inputs, converts clock times to hours, creates a late-bedtime indicator, and scales the model features with `StandardScaler`.

## Technologies and libraries

The repository is Python-only. Imports used by the source code include:

| Technology / library | Use in this project |
| --- | --- |
| Python | Application and model-training code |
| Streamlit | Web interface and interactive inputs |
| pandas | Tabular data loading and transformation |
| NumPy | Numeric processing and synthetic training-data generation |
| scikit-learn | Feature scaling, train/test split, classifiers, and evaluation metrics |
| joblib | Saving and loading the trained model bundle |
| Plotly Graph Objects | Sleep-quality distribution chart |

There is no `requirements.txt`, `pyproject.toml`, or other dependency manifest in the repository, so package versions are not pinned. The interface styling is embedded in `app.py` (CSS and HTML passed to Streamlit); there is no separate HTML template directory. No database configuration or database integration is present in the repository.

## Project structure

```text
.
├── app.py                              # Streamlit interface, prediction flow, and analytics
├── preprocess.py                       # Time parsing, feature engineering, encoding, scaling
├── recommendations.py                  # Rule-based reasons and sleep-habit tips
├── train_model.py                      # Synthetic-data generation, model evaluation, and saving
├── sleep_dataset.csv                   # Dataset file at repository root
├── best_model.pkl                      # Pickled model artifact at repository root
├── preprocess.cpython-314.pyc           # Compiled Python bytecode artifact
├── recommendations.cpython-314.pyc      # Compiled Python bytecode artifact
├── settings.json                       # Editor setting
└── README.md
```

`train_model.py` uses `data/sleep_dataset.csv` and writes the selected model bundle to `models/best_model.pkl`, creating those directories as needed. `app.py` loads the dataset and model from those same `data/` and `models/` paths. The root-level `sleep_dataset.csv` and `best_model.pkl` shown above are not loaded by those paths directly. Running the training script creates the expected paths; when `data/sleep_dataset.csv` is absent, it generates a synthetic dataset. The committed `.pyc` files are compiled artifacts and are not needed to run the Python source.

## Run locally

Use Python with the libraries listed above installed. Since the repository does not pin dependencies, install the imports used by the application and training code in your environment:

```bash
python -m pip install streamlit pandas numpy scikit-learn joblib plotly
```

From the repository root, prepare the model bundle and start the app:

```bash
python train_model.py
streamlit run app.py
```

The training command creates `data/` and `models/` as needed, evaluates the four classifiers, and saves `models/best_model.pkl`. The Streamlit app expects that model bundle to exist and reads analytics data from `data/sleep_dataset.csv`.

## Future enhancements

The following are proposed improvements, not features currently implemented:

- Add a dependency manifest with supported Python and package versions.
- Add automated tests for feature engineering, prediction input handling, and recommendation rules.
- Document the dataset's provenance, schema, and limitations; validate and report model performance on a suitable held-out, non-synthetic dataset.
- Resolve the difference between root-level data/model artifacts and the `data/` and `models/` paths used by the scripts, and exclude generated bytecode from version control.
- Add optional prediction history or user-configurable tracking, with an explicit persistence and privacy design if storage is introduced.
- Improve accessibility and make the interface's explanatory text and recommendation guidance easier to customize.

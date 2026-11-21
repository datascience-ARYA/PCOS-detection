# PCOS Detection Model – Streamlit app

Showcase of `pcos_prediction_classifier.ipynb` (github.com/Fayyez/PCOS-Detection-Model).

## Run
```bash
pip install -r requirements.txt
# put PCOS_data_without_infertility.xlsx (Kaggle: prasoonkottarathil/polycystic-ovary-syndrome-pcos) in ./data/
# ...or just upload it from the app's sidebar
streamlit run app.py
```

## Layout
- `app.py` – UI (navigation, styling, pages)
- `pipeline.py` – the notebook code, ported 1:1 (same parameters, seeds, grids, figure sizes)
- `.streamlit/config.toml` – theme

Model training takes a few minutes the first time (two GridSearchCV runs + stacking); results are cached afterwards.

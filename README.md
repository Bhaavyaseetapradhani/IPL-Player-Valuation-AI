# IPL-Player-Valuation-AI
# IPL ValueAI 

A machine learning dashboard that predicts IPL auction prices based on player stats and identifies undervalued/overvalued players.

## Live Pages
- **Home** — Hero stats, methodology, top value gaps
- **Players** — Search and filter all 42 players
- **Value Gap** — Visual breakdown of over/undervalued players
- **Model** — Feature importance, scatter plots, model metrics (R²=0.87)

## Running Locally
Just open `index.html` in a browser, or serve with any static server:
```bash
npx serve .
# or
python -m http.server 8000
```

## Data Pipeline (optional)
To regenerate data from Cricsheet YAML files:
```bash
pip install pandas scikit-learn pyyaml numpy
python pipeline/process_ipl_data.py
```
Place Cricsheet IPL YAML files in a `cricsheet_data/` folder first.

## Stack
- Vanilla HTML/CSS/JS + Tailwind CDN
- Python + scikit-learn (Gradient Boosting) for the model
- Data from [Cricsheet.org](https://cricsheet.org)

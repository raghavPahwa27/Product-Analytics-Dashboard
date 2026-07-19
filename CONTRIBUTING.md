# Contributing

Thank you for your interest in contributing to this project!

## Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/raghavPahwa27/Product-Analytics-Dashboard.git
cd Product-Analytics-Dashboard

# 2. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Download data and build database
python database.py

# 6. Run the pipeline
python preprocessing.py
python feature_engineering.py
python train.py

# 7. Start the dashboard
streamlit run app.py
```

## Code Standards

- Follow PEP 8
- Keep functions short and focused (< 30 lines where possible)
- Add a one-line docstring to every function
- No unused imports; no dead code
- Run `python3 -m py_compile <file>` before committing

## Project Structure

```
app.py              # Thin router (~70 lines)
views/              # One module per dashboard view
utils/              # Shared utilities (data, ui, ai, pdf)
sql/                # SQL query files
model/              # Saved model artefacts
assets/             # Generated chart images
data/               # Local data (gitignored)
```

## Pull Requests

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes with clear commit messages
3. Ensure `python3 -c "import ast; ast.parse(open('app.py').read())"` passes
4. Open a PR with a description of what changed and why

## Reporting Issues

Open a GitHub Issue with:
- A clear title
- Steps to reproduce
- Expected vs actual behaviour
- Python and Streamlit version

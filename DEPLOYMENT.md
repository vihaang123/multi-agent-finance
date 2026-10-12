# Deployment guide

> **Status (be precise about what has and has not happened):** the app has been run locally with a real Streamlit
> server and exercised in a real (headless Chromium) browser, and the code is pushed to
> <https://github.com/vihaang123/multi-agent-finance>. It has **not** been deployed to Streamlit Community Cloud from
> this environment, because that requires signing in with *your* GitHub account in a browser. There is therefore **no
> live URL yet**. Follow the steps below (about 5 minutes) to get one.

## Option 1 - Streamlit Community Cloud (recommended, free)

1. Open <https://share.streamlit.io> and sign in with the GitHub account that owns `vihaang123/multi-agent-finance`
   (authorise Streamlit to read the repository if asked; if the repo is private, grant access to private repos).
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `vihaang123/multi-agent-finance`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** choose a sub-domain, e.g. `multi-agent-finance`
4. Open **Advanced settings** and choose **Python 3.12 or 3.13** (the code was developed on 3.13).
   No secrets or environment variables are needed.
5. Click **Deploy**. The first build installs `requirements.txt` (a minute or two). Streamlit Cloud then gives you the
   public URL `https://<your-subdomain>.streamlit.app`.

### Verify the live app (2-minute checklist)
- The yellow **Synthetic data** banner is visible on every page.
- **Dashboard** shows KPI cards, a value chart, a donut and a performance table without needing any click.
- **Simulation**: press **Reset**, then **Step** - the step counter goes to 1; press **Start** - it advances until you
  press **Pause**.
- **RL Lab**: leave the defaults but set *Training episodes* to 100 and press **Train from scratch** - a progress bar
  runs and the learning curves appear.
- **Experiments**: the *Precomputed reference* view shows four rows (A-D).

### Things to know on Community Cloud
- Free apps have limited CPU/RAM and go to sleep after inactivity (the first visit afterwards takes ~30 s to wake).
  Training speed there will be roughly 1-2x slower than on a laptop (about 15-30 s per 1,000 episodes of 4-agent
  training locally). The app caps live training at 3,000 episodes and stops a run after 150 s, keeping what was learned.
- Session state lives in each visitor's own session; visitors do not affect each other.
- To update the app: `git push` to `main` - Streamlit Cloud redeploys automatically.

## Option 2 - Run locally

```bash
git clone https://github.com/vihaang123/multi-agent-finance.git
cd multi-agent-finance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py                          # opens http://localhost:8501
```

## Option 3 - Docker (provided, **not built or tested in the authoring environment**)

```bash
docker build -t multi-agent-finance .
docker run -p 8501:8501 multi-agent-finance             # then open http://localhost:8501
```
The `Dockerfile` was written but could not be built where this project was developed (no Docker daemon was available),
so treat it as untested until you have built it once.

## Regenerating the precomputed reference results

`data/reference_results.json` and `data/reference_qtables.npz` come from a real run:

```bash
python -m scripts.build_reference          # ~5 minutes: experiments A-D, 5000 training episodes, 50 held-out episodes
```
`tests/test_charts_and_reference.py` checks that these files really reproduce when the engine is re-run.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: dashboard` / `config` | The main file must be `streamlit_app.py` **at the repository root** (not inside a sub-folder). |
| Build fails on a dependency | In *Advanced settings* choose Python 3.12/3.13; the pinned minimum versions in `requirements.txt` need a recent Python. |
| Reference tab says files not found | Make sure `data/` was committed (it is part of the repo), or run `python -m scripts.build_reference`. |
| Training stops early with a warning | Intentional 150 s time budget; press **Continue training** to go on. |

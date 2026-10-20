# Deploying to Vercel

The app is a Next.js project (front end) with one Python serverless function (`api/index.py`, FastAPI). Vercel detects both
automatically. No environment variables are needed.

> **Status:** deployed at <https://multi-agent-finance-eight.vercel.app/> (Vercel project linked to the GitHub repository `vihaang123/multi-agent-finance`; pushes
> to `main` redeploy automatically). The checklist below was run by hand in Chrome against that URL. Build logs and runtime logs were
> not inspected (the connector used for testing had no access to the Vercel account), so check them in the Vercel dashboard if anything misbehaves.

## Steps (dashboard)

1. Go to <https://vercel.com/new> and sign in with the GitHub account that owns the repo (`vihaang123`).
2. **Import** `vihaang123/multi-agent-finance`. If it is not listed, click *Adjust GitHub App Permissions* and grant access.
3. Leave **Framework Preset = Next.js**, **Root Directory = `./`**, default build/install commands, and add **no** environment variables.
4. Click **Deploy**. Every later `git push` to `main` redeploys automatically.

## Steps (CLI alternative)
```bash
npm i -g vercel
vercel login
vercel link          # create/link the project
vercel --prod
```

## How it is wired
- `vercel.json` gives `api/index.py` a 60 s `maxDuration` and excludes front-end/test folders from the Python bundle.
  (60 s is the Hobby-plan maximum; the app still only runs ~50 episodes, about a few seconds, per request.)
- `next.config.mjs` rewrites `/api/:path*` to the Python function in production. Locally it rewrites to `LOCAL_API_URL`.
- `.python-version` pins Python 3.13; `requirements.txt` lists only the API's dependencies (no matplotlib, no test tools).
- `.vercelignore` keeps `node_modules`, `tests`, `results` and `docs` out of uploads.

## Verify after deploying
1. Open `https://<your-project>.vercel.app/api/health` -> should return `{"status":"ok","synthetic_data":true}`.
2. Open `/api/meta` -> JSON with the agent roles and baskets.
3. Open `/` -> Dashboard with the **synthetic data** banner and KPI cards; try Simulation -> *Step*, and RL Lab -> *Train*
   (200 episodes).
4. If a page shows an API error, open the project in Vercel -> *Logs* (runtime logs) and the failed deployment's *Build Logs*.
5. Optional: run the browser suite against the live URL: `E2E_BASE_URL=https://<your-project>.vercel.app pytest tests/e2e -q`.
   (If *Deployment Protection* is enabled for previews, use the production domain or disable protection for testing.)

## Things to watch (known, unverified on Vercel)
- **Bundle size:** numpy + pandas + plotly + the stored reference (`data/`) must fit Vercel's Python function limit (500 MB
  uncompressed; a few hundred MB at most here). If the build complains, check `excludeFiles` in `vercel.json`.
- **Cold starts:** the first request after idle time imports numpy/pandas/plotly and loads the reference; expect a few seconds.
- **Time limit:** if training chunks time out on a slower CPU, lower the `CHUNK` constant in `app/rl-lab/page.tsx` and `app/experiments/page.tsx` or raise
  `maxDuration` on a paid plan.
- **Statelessness:** the server keeps nothing. Trained agents are saved in the visitor's own browser (localStorage); *Reset session* in the sidebar clears them.

## Local development
```bash
pip install -r requirements.txt && npm install
uvicorn api.index:app --port 8000
LOCAL_API_URL=http://127.0.0.1:8000 npm run dev      # http://localhost:3000
```

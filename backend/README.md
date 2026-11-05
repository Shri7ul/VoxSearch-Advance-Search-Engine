# VoxSearch — Backend

FastAPI backend for VoxSearch.

> **Scope:** Application initialization, centralized settings, logging, and a
> `GET /health` endpoint (Goal 02), running on a reproducible Python 3.12
> environment (Goal 03). Supabase, PostgreSQL, embeddings, transcript
> processing, and search are **not** implemented yet.

## Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app factory, metadata, lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # pydantic-settings configuration
│   │   └── logging.py     # centralized logging setup
│   └── api/
│       ├── __init__.py
│       └── health.py      # GET /health
├── tests/
│   └── test_health.py     # pytest suite
├── .venv/                 # local virtual environment (git-ignored, NOT committed)
├── requirements.txt       # exact direct-dependency pins
├── pytest.ini
├── .env.example           # copy to .env (not committed)
└── README.md
```

Run all commands below **from the `backend/` directory**.

---

## 1. Python 3.12 prerequisite

**Python 3.12 is the project baseline.** Python 3.14 passed the early backend
tests but is *not* assumed suitable for the future ML stack, so do not use it
for this environment.

Check the installed interpreters:

```powershell
py --list
py -0p
```

Confirm a `3.12` entry is present, then verify:

```powershell
py -3.12 --version
python --version
```

> If Python 3.12 is not installed, install a standard python.org / winget
> Python 3.12 first. **Do not substitute 3.13 or 3.14.**

## 2. Virtual environment (Windows PowerShell)

Create the environment **with Python 3.12** (one time):

```powershell
py -3.12 -m venv .venv
```

If `py -3.12` is not registered but a Python 3.12 interpreter exists elsewhere,
point at that interpreter directly instead:

```powershell
& "C:\path\to\Python312\python.exe" -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session
only (no permanent policy change):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Confirm activation — `python --version` must report **3.12.x** and
`python -c "import sys; print(sys.prefix)"` must point at `.venv`.

To deactivate:

```powershell
deactivate
```

> On macOS / Linux use `source .venv/bin/activate` instead.
> `.venv/` is git-ignored — **never commit the virtual environment**.

## 3. Dependency installation

With `.venv` activated:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the six direct dependencies exactly for reproducible
installs:

| Package | Pin |
| --- | --- |
| `fastapi` | 0.162.0 |
| `uvicorn[standard]` | 0.54.0 |
| `pydantic-settings` | 2.16.2 |
| `python-dotenv` | 1.2.4 |
| `pytest` | 9.2.0 |
| `httpx` | 0.28.1 |

No ML packages, PyTorch, Transformers, Whisper, database drivers, or other
future dependencies are included in this goal.

Verify integrity:

```powershell
python -m pip check
python -m pip list
```

`pip check` must report `No broken requirements found.`

## 4. Environment configuration

Settings are loaded by `pydantic-settings` from environment variables and,
optionally, a local `.env` file. All values have safe defaults, so the app runs
with **no configuration at all**.

```powershell
copy .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `voxsearch-backend` | Service name (`/health` "service" field, OpenAPI title) |
| `APP_ENV` | `development` | Runtime environment label |
| `APP_VERSION` | `0.1.0` | Version exposed by `/health` and OpenAPI |
| `API_PREFIX` | `/api/v1` | Prefix reserved for future versioned routers |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |

`.env` is git-ignored. **Never put credentials or secrets in `.env.example`.**

## 5. Running tests

With `.venv` activated:

```powershell
pytest -v
```

The suite uses FastAPI's `TestClient` and covers:

- `GET /health` returns HTTP 200.
- The response contains the expected fields and payload.
- Application metadata (title, version, description, API prefix) is correct.

All four tests must pass.

## 6. Starting the FastAPI server

With `.venv` activated:

```powershell
uvicorn app.main:app --reload
```

By default this serves `http://127.0.0.1:8000`. To bind explicitly:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

Verify the health endpoint (from a second terminal):

```powershell
curl http://127.0.0.1:8000/health
```

Expected `200 OK` response:

```json
{
  "status": "healthy",
  "service": "voxsearch-backend",
  "version": "0.1.0"
}
```

---

## Validation performed

### Goal 03 — reproducible Python environment

- `backend/.venv` created from CPython **3.12.14** (a `conda-forge` build in an
  existing conda env), because `py -3.12` is not registered on this machine.
  No new Python was installed and 3.13/3.14 were not used.
- `python --version` → `Python 3.12.14`
- `python -m pip check` → `No broken requirements found.`
- `python -m pip list` → 29 packages, clean resolution
- `pytest -v` → **4 passed**

### Goal 02 — backend foundation

- `pip install -r requirements.txt`
- `pytest -v`
- `uvicorn app.main:app --host 127.0.0.1 --port 8000` + `curl /health`

See the Goal 02/03 final reports in the session history for recorded results.

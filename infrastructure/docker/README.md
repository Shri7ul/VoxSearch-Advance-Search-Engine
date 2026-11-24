# Goal 47 — Production Docker setup

Reproducible, local production-like deployment artifacts. **No Docker build or
run was executed as part of this goal**; no live infrastructure was contacted.

## Files

- `backend.Dockerfile` — Python 3.12.10-slim, installs the fully resolved
  `backend/requirements-lock.txt`, non-root user, `/health` healthcheck.
- `frontend.Dockerfile` — Node 20.19.5, `npm ci` from the committed
  `frontend/package-lock.json`, multi-stage build.
- `docker-compose.yml` — API, web and Redis services with healthchecks and a
  persistent Redis volume.
- `../../.dockerignore` — excludes secrets, virtualenvs, build caches and `.env`.

## Usage (requires a local Docker installation)

```
cd infrastructure/docker
docker compose up --build
```

Secrets come only from the git-ignored `backend/.env` at runtime. The image
contains no credentials. `NEXT_PUBLIC_API_BASE_URL` is a public browser-side
configuration value, not a secret.

## Known limitations

- No image was built or started here; Docker is not available in this
  environment.
- Pinned version tags improve repeatability but are not digest-pinned; upstream
  tags can be retagged. Full bit-for-bit reproducibility is therefore NOT
  claimed.
- `environment` values in Compose are configuration, not verified runtime
  behavior.
- The frontend dependency lockfile reports npm audit advisories; resolving
  those is intentionally out of scope for Goal 47.
- No production database migration or external service access is performed.

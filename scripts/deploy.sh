#!/usr/bin/env bash
# Deploy Cloud Run service "topology" from source (Cloud Build builds the Dockerfile).
# One revision serves both front doors: https://topology-cnc.web.app/ (Hosting ** -> Cloud Run)
# and https://eisensoftware.web.app/topology/ (platform Hosting rewrite /topology{,/**}).
# GET /health then reports the APP_VERSION, APP_COMMIT and APP_DEPLOYED_AT set here.
#
#   scripts/deploy.sh                               # deploy and send it all traffic
#   scripts/deploy.sh --no-traffic --tag platform   # extra args are passed to `gcloud run deploy`
#   DRY_RUN=1 scripts/deploy.sh ...                 # print the command, deploy nothing
#
# --update-env-vars (never --set-env-vars) keeps the other variables already set on the service.
set -euo pipefail

cd "$(dirname "$0")/.."

commit=$(git rev-parse HEAD)
if tag=$(git describe --tags --exact-match 2>/dev/null); then
  version=${tag#v}
else
  version="0.0.0+sha.${commit:0:12}"
fi
deployed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ -n "$(git status --porcelain)" ]; then
  echo "warning: uncommitted changes will be deployed, but APP_COMMIT will say $commit" >&2
fi

cmd=(gcloud run deploy topology --source . --region us-central1 --project researcher-455022
  --update-env-vars "APP_VERSION=$version,APP_COMMIT=$commit,APP_DEPLOYED_AT=$deployed_at" "$@")

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi
exec "${cmd[@]}"

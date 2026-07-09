# ──────────────────────────────────────────────────────────────────────────
# Convenience targets for both container backends this project supports:
#
#   Linux / MX Linux  -> Docker Compose   (docker-compose.yml, `docker` targets)
#   macOS / Apple Sil -> Apple `container`(container/*.sh,     `mac`    targets)
#
# See docs/Apple_Container_Setup.md for the macOS install/setup story.
# ──────────────────────────────────────────────────────────────────────────

IMAGE_TAG := wc26predictor:latest

.PHONY: build up down logs status \
        mac-build mac-up mac-down mac-deploy mac-restart mac-logs mac-status mac-clean

# ── Linux (Docker Compose) ───────────────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f api

status:
	docker compose ps

# ── macOS (Apple `container` CLI) ────────────────────────────────────────────
mac-build:
	./container/build.sh $(IMAGE_TAG)

mac-up:
	./container/up.sh $(IMAGE_TAG)

mac-down:
	./container/down.sh

# Full first-run / after-Dockerfile-change flow: build then (re)start.
mac-deploy: mac-build mac-up

mac-restart: mac-down mac-up

mac-logs:
	container logs -f wc26predictor

mac-status:
	container list --all
	@curl -fsS "http://localhost:$${API_PORT:-8000}/docs" -o /dev/null \
	    && echo "api: healthy (docs endpoint reachable)" \
	    || echo "api: not reachable on port $${API_PORT:-8000}"

# Stop + remove the container AND delete the pytensor_cache volume.
mac-clean:
	./container/down.sh --volume
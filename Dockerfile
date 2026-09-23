# ---- frontend ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- app ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    TOPOLOGY_CACHE=/var/cache/topology \
    PORT=8080
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY topology/ topology/
RUN pip install --no-cache-dir .
# Bake the state boundaries into the image so cold starts don't hit the Census server.
RUN python -c "from topology.boundary import load_states; load_states()" \
    && rm -f /var/cache/topology/*.zip && chmod -R a+rwX /var/cache/topology
COPY --from=web /web/dist web/dist
# Installed package lives in site-packages; point the server at the built frontend.
ENV TOPOLOGY_WEB_DIST=/app/web/dist
EXPOSE 8080
CMD ["sh", "-c", "topology serve --host 0.0.0.0 --port ${PORT}"]

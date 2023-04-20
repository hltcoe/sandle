FROM node:16 AS builder

COPY . /app
RUN cd /app && \
    rm -rf dist && \
    npm ci && \
    npm run build


FROM nginx:1.22

# Configure server
COPY nginx.default.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
COPY 50x.json /usr/share/nginx/html/v1/

# Sentry build-time configuration
ARG VITE_SENTRY_DSN
ARG VITE_SENTRY_RELEASE

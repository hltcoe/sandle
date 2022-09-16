FROM nginx:1.22


# Install build tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y npm && \
    rm -rf /var/lib/apt/lists/*

# Configure server
COPY nginx.default.conf /etc/nginx/conf.d/default.conf

# Sentry build-time configuration
ARG VITE_SENTRY_DSN
ARG VITE_SENTRY_RELEASE

# Build and deploy app
COPY . /tmp/app-source
RUN cd /tmp/app-source && \
    rm -rf dist && \
    npm ci && \
    npm run build && \
    mv dist/* /usr/share/nginx/html/ && \
    cd / && \
    rm -rf /tmp/app-source

# Clean up build tools
RUN apt-get purge -y npm && \
    apt-get autoremove -y

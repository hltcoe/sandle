# openaisle
Run a clone of OpenAI's API Service in your Local Environment ("OpenAISLE").

## Setup

This repository provides the following Docker services:

 * `opt`: a service that implements a subset of [the OpenAI `/v1/completions` API](https://beta.openai.com/docs) (without authentication) using a single-threaded web server on top of OPT.  Alone, this service is suitable for [a single user at a time](#serving-the-api-for-a-single-user-without-docker).
 * `openai-adapter`: a service that implements a subset of [the OpenAI `/v1/models`, `/v1/models/<model>`, and `/v1/completions` APIs](https://beta.openai.com/docs) (with authentication), implementing the latter by calling the `opt` service.  It uses a multi-threaded web server and is suitable for multiple users.
 * `demo`: an [nginx](https://nginx.org) web server that acts as a reverse proxy in front of the API (the `openai-adapter` service) and serves a web interface for text completion using the proxied API.

These services can be run together on your local machine using [Docker Compose](https://docs.docker.com/compose/).  The Docker Compose configuration file (`docker-compose.yml`) specifies how they are invoked.

### Demo

A web interface is provided to illustrate the API functionality and facilitate exploration of the models:

<img width="888" alt="Screen Shot of demo web interface" src="https://user-images.githubusercontent.com/457238/178364339-6d62e7b7-9687-46ed-9312-9e5cb1446705.png">

To build and run this demo using Docker Compose, do:

```bash
docker-compose up --build
```

By default, the demo web interface and API endpoint will be bound to port 80 on the host.  Go to
`http://localhost` in your browser to use the web interface.

#### BRTX

The docker compose version installed on BRTX is outdated and does not
work with our docker compose configuration file.  To use docker compose
on BRTX, [install a new, standalone version of docker
compose](https://docs.docker.com/compose/install/compose-plugin/#install-the-plugin-manually)
to your home directory and run that version instead of the
system-installed version.

#### K80

The Nvidia Tesla K80 is no longer actively supported by Nvidia and
[newer versions of the `nvidia/cuda` Docker base image are configured
not to run on K80 cuda drivers](https://gitlab.com/nvidia/container-images/cuda/-/issues/165#note_1005164251).  To work around this, modify the docker
compose configuration to add the desired cuda driver version to the
`NVIDIA_REQUIRE_CUDA` environment variable in the `opt` service.  For
example:

```yaml
services:
  opt:
    environment:
      - 'NVIDIA_REQUIRE_CUDA=cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450'
```

### Serving the API for a single user without Docker

If you only need the API for a single user, it is easy to run the `opt` service by itself, outside of Docker.  Ensure the cuda toolkit and pytorch are installed, then install the Python requirements specified in `opt/requirements.txt`, and run (for example)

```bash
python opt/serve-opt.py --port 12349
```

to serve the partial `/v1/completions` API on port 12349 on your local host.  The equivalent Docker usage would be (approximately):

```bash
docker build -t $USER/opt opt && docker run -it -p 12349:8000 $USER/opt --port 8000
```

## Usage

### Authentication

By default, the openai-adapter service will generate a random API key every time it starts up.
This API key will be logged to the console.  You can also specify your own (base-64–encoded) API
key by passing the `--auth-token` argument on the command line.

[As in the OpenAI API](https://beta.openai.com/docs/api-reference/authentication), the API key can be used either as a "Bearer" authentication token or as a basic authentication password (with the user being the empty string).

### Example API calls

The following command may be called against OpenAI's service or against a local OpenAisle service.
For example, on OpenAI:  

```bash
curl "https://api.openai.com/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
  "model": "text-davinci-002",
  "prompt": "Say this is a test"
}'
```

and on a local OpenAisle deployment:

```bash
curl "http://YOUR_OPENAISLE_SERVER/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
  "model": "facebook/opt-2.7b",
  "prompt": "Say this is a test"
}'
```

Note that OpenAisle only supports HTTP (not HTTPS) at this time.

### API Documentation

See our [API documentation](https://hltcoe.github.io/openaisle) for a description of the subset of the OpenAI API implemented by OpenAisle.
This documentation is generated using the Swagger UI on our API definition file at `docs/swagger.yaml`.

## Testing

Example runtime test using the Apache Bench tool (installed by default on OS X):

```
ab -n 10 -c 1 -s 60 -p qa.txt -T application/json -A :YOUR_API_KEY -m POST http://YOUR_OPENAISLE_SERVER/v1/completions
```

where `qa.txt` is a text file in the current directory that contains the prompt JSON (for example, `{"model": "facebook/opt-2.7b", "prompt": "Say this is a test"}`).

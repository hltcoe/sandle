# Sandle

[![Docker build status](https://github.com/hltcoe/sandle/actions/workflows/docker-build.yml/badge.svg)](https://github.com/hltcoe/sandle/actions/workflows/docker-build.yml)
[![Python test status](https://github.com/hltcoe/sandle/actions/workflows/python-test.yml/badge.svg)](https://github.com/hltcoe/sandle/actions/workflows/python-test.yml)
[![License](https://img.shields.io/badge/License-BSD-blue)](https://github.com/hltcoe/sandle/blob/main/LICENSE)

Run a large language modeling SANDbox in your Local Environment (SANDLE).

This repository provides a Docker Compose system for hosting and interacting with large language models on your own hardware.  It includes a web sandbox:

![Screen Shot 2022-08-09 at 1 29 33 PM](https://user-images.githubusercontent.com/457238/183720063-9c87ce24-e4d4-4a9d-b883-b085a12f48a8.png)

and an OpenAI-like REST API:

![Screen Shot 2022-08-09 at 1 14 44 PM](https://user-images.githubusercontent.com/457238/183715419-56c1467f-e5fe-4ebe-9c3f-b1feb7c4e9b9.png)


## Setup

To build and run SANDLE with default settings using Docker Compose, do:

```bash
docker-compose up --build
```

By default, the demo web interface and API endpoint will be bound to port 80 on the host.  Go to
`http://localhost` in your browser to use the web interface.

#### BRTX

The Docker Compose version installed on BRTX is older and does not
work with our configuration file, which requires Docker Compose
v1.28.0 or later.  To use Docker Compose
on BRTX, [install a new, standalone version of docker
compose](https://docs.docker.com/compose/install/compose-plugin/#install-the-plugin-manually)
to your home directory and run that version instead of the
system-installed version.

Additionally, on BRTX, the server will be bound to the local host using IPv4 but `localhost`
will resolve to the local host using IPv6.  When connecting to the API, specify `127.0.0.1` or
`localhost4` instead of `localhost`.

#### K80

The Nvidia Tesla K80 is no longer actively supported by Nvidia and
[newer versions of the `nvidia/cuda` Docker base image are configured
not to run on K80 cuda drivers](https://gitlab.com/nvidia/container-images/cuda/-/issues/165#note_1005164251).  To work around this, modify the docker
compose configuration to add the desired cuda driver version to the
`NVIDIA_REQUIRE_CUDA` environment variable in the `backend-hf` service.  For
example:

```yaml
services:
  backend-hf:
    environment:
      - 'NVIDIA_REQUIRE_CUDA=cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450'
```


## Usage

### Authentication

By default, the openai-wrapper service will generate a random API key every time it starts up.
This API key will be logged to the console.  You can also specify your own (base-64–encoded) API
key by passing the `--auth-token` argument on the command line.

[As in the OpenAI API](https://beta.openai.com/docs/api-reference/authentication), the API key can be used either as a "Bearer" authentication token or as a basic authentication password (with the user being the empty string).

### Example API calls

The following command may be called against OpenAI's service or against a local Sandle service.
For example, on OpenAI:  

```bash
curl "https://api.openai.com/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -d '{
  "model": "text-davinci-002",
  "prompt": "Say this is a test"
}'
```

and on a local Sandle deployment:

```bash
curl "http://YOUR_SANDLE_SERVER/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_SANDLE_API_KEY" \
  -d '{
  "model": "facebook/opt-2.7b",
  "prompt": "Say this is a test"
}'
```

Note that Sandle only supports HTTP (not HTTPS) at this time.

### API Documentation

See our [API documentation](https://hltcoe.github.io/sandle) for a description of the subset of the OpenAI API implemented by Sandle.
This documentation is generated using the Swagger UI on our API definition file at `docs/swagger.yaml`.


## Advanced Usage


This repository provides the following Docker services:

 * `backend-hf`: a service that implements a subset of [the OpenAI `/v1/completions` API](https://beta.openai.com/docs) (without authentication) using a single-threaded web server on top of HuggingFace, supporting `OPT` and `Bloom` at the time of writing.  Alone, this service is suitable for [a single user at a time](#serving-the-api-for-a-single-user-without-docker).
 * `openai-wrapper`: a service that implements a subset of [the OpenAI `/v1/models`, `/v1/models/<model>`, and `/v1/completions` APIs](https://beta.openai.com/docs) (with authentication), implementing the latter by calling the `backend-hf` service.  It uses a multi-threaded web server and is suitable for multiple users.
 * `demo`: an [nginx](https://nginx.org) web server that acts as a reverse proxy in front of the API (the `openai-wrapper` service) and serves a web interface for text completion using the proxied API.

These services can be run together on your local machine using [Docker Compose](https://docs.docker.com/compose/), configured in `docker-compose.yml`.

### Serving the API for a single user without Docker

If you only need the API for a single user, you can run the `backend-hf` service by itself, outside of Docker.  Ensure the cuda toolkit and pytorch are installed, then install the Python requirements specified in `backend-hf/requirements.txt`, and run (for example)

```bash
python backend-hf/serve-backend-hf.py --port 12349
```

to serve the partial `/v1/completions` API on port 12349 on your local host.  The equivalent Docker usage would be (approximately):

```bash
docker build -t $USER/backend-hf backend-hf && docker run -it -p 12349:8000 $USER/backend-hf --port 8000
```


## Development

To set up a development environment for the demo web interface, install a recent version of `npm`, go to the `demo` subdirectory, and do:

```
npm install
```

Then configure your development app by copying `.env.development` to `.env.development.local` and changing the values set in the file accordingly.  In particular, make sure you set `VUE_SANDLE_HOST` and `VUE_SANDLE_PORT` to the host and port of the API implementation you are using for development.  The demo service acts as a simple reverse proxy for the API implementation provided by the openai-wrapper service, so if you wish to run an API implementation yourself, you can run `docker-compose up` as usual, then use `localhost` and the demo service port bound to your host machine (by default, port 80) as your host and address.

Once you've done that, you can start a development web server with:

```
npm run dev
```

Note that in the case you used `docker-compose up` to provide an API implementation, you will now have two versions of the demo interface running: one on port 80 (by default), running from the demo service container, and one on port 3000 (by default), running via `npm` directly on your host machine.

### Stubbing out the backend

If you cannot or do not wish to run a full language model backend during testing and development, you may use the stub backend instead.  This backend provides a type-compliant implementation of the `/v1/completions` API that returns a fixed completion of `" world!"`, as if responding to the prompt `"Hello,"`.  Cute, right?

Using this backend requires a small amount of additional setup.  First, start the development server as usual:

```
npm run dev
```

Then, instead of running `docker-compose up`, launch the `openai-wrapper` service standalone (binding to whichever port the frontend is configured to connect to, here 54355, on your local machine):

```
docker-compose build openai-wrapper && docker-compose run --rm -p 54355:8000 --no-deps openai-wrapper
```

Finally, do the following to build and run the stub backend on the default network created by Docker Compose:

```
docker run --network sandle_default --name backend-hf --rm `docker build -q backend-stub`
```


## Testing

### Fuzz Testing

To perform fuzz testing using the Microsoft RESTler tool in Docker:

First, bring up the Sandle system using the test authentication
token:

```
SANDLE_AUTH_TOKEN=dGVzdA== docker-compose up --build
```

Then, run `run-fuzz-test-docker.bash` to build the `restler` Docker
image if it does not exist and run RESTler on the API
specification in `docs/swagger.yaml`:

```
bash run-fuzz-test-docker.bash
```

This script will create the directory `fuzz-test-output`, bind it to
the RESTler Docker container, and write the output for each step of the
testing procedure to the appropriately named subdirectory of
`fuzz-test-output`.  Additionally, at the end of each step, the
contents of `fuzz-test-output/STEP/ResponseBuckets/runSummary.json`
(with `STEP` replaced with the step name) will be printed to the
console.  If after any step the number of failures reported in that
file is greater than zero, the test procedure will terminate.

### Benchmarking

Example runtime test using the Apache Bench tool (installed by default on OS X):

```
ab -n 10 -c 1 -s 60 -p qa.txt -T application/json -A :YOUR_API_KEY -m POST http://YOUR_SANDLE_SERVER/v1/completions
```

where `qa.txt` is a text file in the current directory that contains the prompt JSON.  Example file contents:

```json
{"model": "facebook/opt-2.7b", "prompt": "Say this is a test"}
```

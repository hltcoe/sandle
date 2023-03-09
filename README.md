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

To build and run SANDLE with the HuggingFace backend using Docker Compose, do:

```bash
cp docker-compose.backend-hf.yml docker-compose.override.yml
docker-compose up --build
```

By default, the demo web interface and API endpoint will be bound to port 80 on the host.  Go to
`http://localhost` in your browser to use the web interface.

You must have an API key to use the web interface or API endpoint; by default, one will be
generated and logged on startup.  If you wish to specify the accepted API key explicitly instead
of using a randomly generated key, set the `SANDLE_AUTH_TOKEN` environment variable with the
desired API key when running `docker-compose`:

```
SANDLE_AUTH_TOKEN=ExampleAPIKey docker-compose up --build
```

If you wish to limit the models that can be used—perhaps you want to support a particularly
large model and don't want to incur the overhead of loading it into memory more than once—then
set the `SANDLE_SINGLE_MODEL` environment variable with the desired model name when running
`docker-compose`:

```
SANDLE_SINGLE_MODEL=bigscience/bloom docker-compose up --build
```

#### BRTX

The Docker Compose version installed on BRTX is older and does not
work with our configuration file, which requires Docker Compose
v1.28.0 or later.  To use Docker Compose
on BRTX, [install a new, standalone version of docker
compose](https://docs.docker.com/compose/install/compose-plugin/#install-the-plugin-manually)
to your home directory and run that version instead of the
system-installed version.  For example, to download Docker Compose
standalone version 2.7.0:

```bash
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o docker-compose
chmod 755 docker-compose
./docker-compose --version
```

Additionally, on BRTX, the server will be bound to the local host using IPv4 but `localhost`
will resolve to the local host using IPv6.  When connecting to the API, specify `127.0.0.1` or
`localhost4` instead of `localhost`.


## Usage

### Authentication

API keys the application should accept can be specified in a file, as command line arguments,
or in an environment variable.  If no API keys are specified (the default), one will be
generated and logged on startup.

[As in the OpenAI API](https://beta.openai.com/docs/api-reference/authentication), an API key can be used either as a "Bearer" authentication token or as a basic authentication password (with the user being the empty string).

For more information about specifying API keys, run the following:

```
docker-compose run --no-deps openai-wrapper --help
```

### Example API calls

Calling OpenAI's service is similar to calling a Sandle service.  An example call to OpenAI:

```bash
curl "https://api.openai.com/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -d '{
  "model": "text-davinci-002",
  "prompt": "Say this is a test"
}'
```

and an equivalent call to a Sandle service:

```bash
curl "http://YOUR_SANDLE_SERVER/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer YOUR_SANDLE_API_KEY" \
  -d '{
  "model": "facebook/opt-2.7b",
  "prompt": "Say this is a test"
}'
```

Note that Sandle only comes with support for HTTP, not HTTPS.  If you need HTTPS but don't have a certificate, you can set up a reverse proxy in front of Sandle using [certbot](https://certbot.eff.org).

### API Documentation

See our [API documentation](https://hltcoe.github.io/sandle) for a description of the subset of the OpenAI API implemented by Sandle.
This documentation is generated using the Swagger UI on our API definition file at `docs/swagger.yaml`.


## Advanced Usage


This repository provides the following Docker services:

 * Backend services that implement a subset of [the OpenAI `/v1/completions` API](https://beta.openai.com/docs) without authentication.  These services use single-threaded web servers and are suitable for [one user at a time](#serving-the-api-for-a-single-user-without-docker).
   * `backend-hf`: a backend on top of HuggingFace, supporting models like `OPT` and `Bloom` from the HuggingFace Hub.
   * `backend-llama`: a backend on top of LLaMA.
   * `backend-stub`: a stub backend for development and testing.
 * `openai-wrapper`: a service that implements a subset of [the OpenAI `/v1/models`, `/v1/models/<model>`, and `/v1/completions` APIs](https://beta.openai.com/docs), delegating to backend services accordingly.  This service uses a multi-threaded web server and is suitable for multiple users.
 * `demo`: a web server that provides as a reverse proxy in front of the `openai-wrapper` service as well as a web interface that uses the proxied API.

These services can be run together on your local machine using [Docker Compose](https://docs.docker.com/compose/).  By
default, Docker Compose will load configuration from `docker-compose.yml` and, if it is present,
`docker-compose.override.yml`.  Alternatively, configuration files may be explicitly specified on the command line.
For example, the following command starts Sandle with the HuggingFace backend by specifying the configuration files
explicitly (instead of implicitly, as demonstrated at the beginning of this document):

```bash
docker-compose -f docker-compose.yml -f docker-compose.backend-hf.yml up --build
```

Any number of configuration files can be specified at once as long as their contents can be merged together.  For
example, to start Sandle with both the HuggingFace and the LLaMA backend:

```bash
docker-compose -f docker-compose.yml -f docker-compose.backend-hf.yml -f docker-compose.backend-llama.yml up --build
```

### Serving the API for a single user without Docker

If you only need the API for a single user, you can run a backend service by itself, outside of Docker.  Ensure the appropriate dependencies are installed, then run (for example, using the HuggingFace backend):

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
npm ci
```

Then configure your development app by copying `.env.development` to `.env.development.local` and changing the values set in the file accordingly.  In particular, make sure you set `VITE_SANDLE_URL` to the URL of the API implementation you are using for development.  The `demo` service acts as a simple reverse proxy for the API implementation provided by the openai-wrapper service, so if you wish to run an API implementation yourself, you can run `docker-compose up` as usual, then use `http://localhost` as the URL.

**Note:**  By default, the demo service port is bound to port `80` on the host system.  If this port is in use or if
you don't have access to it, you may need to override it.  To do so, add the `SANDLE_DEMO_PORT` variable to your environment with the desired port as its value, adjust `VITE_SANDLE_URL` in `.env.development.local` accordingly, and then run `docker-compose up` as usual.

Once you've done that, you can start a development web server with:

```
npm run dev
```

This server will run on port 3000 by default and hot-reload the UI when any source files change.

### Stubbing out the backend

If you cannot or do not wish to run a full language model backend during testing and development, you may use the stub backend instead.  To do so, just use the stub backend configuration file in lieu of other backend configuration:

```
docker-compose -f docker-compose.yml -f docker-compose.backend-stub.yml up --build
```


## Testing

### Static Analysis

We use flake8 to automatically check the style and syntax of the code
and mypy to check type correctness.  To perform the checks, go into
a component subdirectory (for example, `backend-hf` or `openai-wrapper`)
and do:

```
pip install -r dev-requirements.txt
flake8
mypy
```

These checks are run automatically for each commit by GitHub CI.

### Property Testing

We use [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to
randomly generate test cases for the backend and assert properties of
interest for the output.  For example, for any valid input, a basic
property that we would like to test is that Sandle doesn't crash on
that input.  A slightly more advanced property might be that the output
does not exceed the user-specified length limit.

Property tests are defined in `backend-hf/tests/test_service.py` and
automatically discovered and run by pytest.

To run the tests, first go to the `backend-hf` subdirectory.  The rest
of this section assumes you are in that directory.

Then, install the basic test requirements:

```
pip install -r dev-requirements.txt
```

The tests assume a backend service exists at http://localhost:8000; you
must start this service yourself.  You can start the service in Docker
or directly on the host machine, depending on your needs.  The
following two examples illustrate how to use these methods to start the
backend service listening to port 8000 and using the first GPU on your
host system.

To start the service in Docker (publishing container port 8000 to host
port 8000):

```
docker build -t backend-hf . \
  && docker run --rm -it --gpus device=0 -p 8000:8000 backend-hf
```

Alternatively, to start the service directly on your host, install
the requirements (CUDA, PyTorch, and the requirements specified in
`requirements.txt`), then run:

```
CUDA_VISIBLE_DEVICES=0 python serve-backend-hf.py
```

Then, you can test that the service is up:

```
curl "http://127.0.0.1:8000/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "facebook/opt-125m",
  "prompt": "Say this is a test"
}'
```

Finally, to run the explicit property test cases:

```
pytest --hypothesis-profile explicit
```

Alternatively, to run explicit test cases and automatically generate
and test new cases (may take a while):

```
pytest
```

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

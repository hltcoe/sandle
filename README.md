# openaisle
Run a clone of OpenAI's API Service in your Local Environment ("OpenAISLE").

## Setup

This repository includes:

 * `opt`: a service that implements a subset of [the OpenAI `/v1/completions` API](https://beta.openai.com/docs) (without authentication) using a single-threaded web server on top of OPT.  Alone, this service is suitable for a single user at a time.
 * `openai-adapter`: a service that implements a subset of [the OpenAI `/v1/models`, `/v1/models/<model>`, and `/v1/completions` APIs](https://beta.openai.com/docs) (with authentication), implementing the latter by calling the `opt` service.  It uses a multi-threaded web server and is suitable for multiple users.
 * `demo`: a web interface for text completion that uses the API and Docker configuration for an [nginx](https://nginx.org) web server that serves the interface and acts as a reverse proxy in front of the API (the `openai-adapter` service).
 * a Docker Compose configuration file (`docker-compose.yml`) that facilitates running these services together and binds the address to the nginx web server to port 80 (by default) on the local machine.

### Demo

A web interface is provided to illustrate the API functionality and facilitate exploration of the models:

<img width="888" alt="Screen Shot of demo web interface" src="https://user-images.githubusercontent.com/457238/178364339-6d62e7b7-9687-46ed-9312-9e5cb1446705.png">

A [Docker Compose](https://docs.docker.com/compose/) configuration is provided to run the demo.  [npm](https://npmjs.com) is used to build the web interface, which runs on the [Vue](https://vuejs.org) web framework.  Because Docker Compose looks in parent
directories for the `docker-compose.yml` file, you can build the interface and run the demo in one
line.  Change directories to the `demo` subdirectory and do:

```bash
npm install && npm run build && docker-compose up --build
```

By default, the demo web interface and API endpoint will be bound to port 80 on the host.  Go to
`http://localhost` in your browser to use the web interface.

## Authentication

By default, the openai-adapter service will generate a random API key every time it starts up.
This API key will be logged to the console.  You can also specify your own (base-64–encoded) API
key by passing the `--auth-token` argument on the command line.

## Usage

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

### Documentation

See our [API documentation](https://hltcoe.github.io/openaisle) for a description of the subset of the OpenAI API implemented by OpenAisle.
This documentation is generated using the Swagger UI on our API definition file at `docs/swagger.yaml`.

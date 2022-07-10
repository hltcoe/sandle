# openaisle
Run a clone of OpenAI's API Service in your Local Environment

## Setup

A Docker Compose configuration is provided to run the demo.  Because Docker Compose looks in parent
directories for the `docker-compose.yml` file, you can build the interface and run the demo in one
line.  Change directories to the `demo` subdirectory and do:

```bash
npm install && npm run build && docker-compose up --build
```

By default, the demo web interface and API endpoint will be bound to port 80 on the host.  Go to
`http://localhost` in your browser to use the web interface.

## Authentication

By default, the demo will generate a random API key every time it starts up.  This API key will
be logged to the console.  You can also specify your own (base-64--encoded) API key by passing the
`--auth-token` parameter to the `openai-adapter` service.

## Example API calls

The following command may be called against OpenAI's service or against a local OpenAisle service
by setting `OPENAISLE_SERVER`, `OPENAISLE_API_KEY`, and `OPENAISLE_MODEL` appropriately.  To use
OpenAI's service, set `OPENAISLE_SERVER` to `api.openai.com`.

```bash
curl "https://$OPENAISLE_SERVER/v1/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENAISLE_API_KEY" \
  -d '{
  "model": "'"$OPENAISLE_MODEL"'",
  "prompt": "Say this is a test"
}'
```
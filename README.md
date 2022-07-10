# openaisle
Run a clone of OpenAI's API Service in your Local Environment

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
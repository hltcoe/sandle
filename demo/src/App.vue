
<template>
  <div class="container my-5">
    <header class="my-4">
      <h1>HuggingFace OPT API Demo &#x1F495;</h1>
      <h6 class="text-muted">
        This web app uses a subset of the OpenAI API to communicate with a local
        deployment of OPT on HuggingFace.
      </h6>
    </header>

    <a
      class="github-fork-ribbon position-absolute position-top-0 position-end-0"
      href="https://github.com/hltcoe/openaisle"
      data-ribbon="Fork me on GitHub"
      title="Fork me on GitHub"
      >Fork me on GitHub</a
    >

    <main>
      <div class="row">
        <div class="col-8">
          <form class="my-3" @submit.prevent="onSubmit(text)">
            <div class="my-3">
              <label class="form-label"
                >Enter some text, then click "Submit" to generate a
                completion.</label
              >
              <textarea
                class="form-control"
                rows="10"
                placeholder="Say this is a test."
                v-model="text"
              />
            </div>
            <div class="my-3 alert alert-danger" v-if="completionsAlert">
              {{ completionsAlert }}
            </div>
            <div class="my-3">
              <button
                v-if="!runningCompletions"
                type="submit"
                class="btn btn-primary"
              >
                Submit
              </button>
              <button v-else type="submit" class="btn btn-primary disabled">
                Working
              </button>
            </div>
          </form>
        </div>
        <div class="col-4">
          <div class="m-3">
            <label class="form-label" for="api-key-input">
              API key
            </label>
            <div class="d-flex">
              <input
                type="text"
                class="form-control flex-grow-1"
                id="api-key-input"
                v-model="apiKey"
              />
              <i
                class="bi fs-4 ps-1"
                :class="{
                  'bi-check-lg': models,
                  'text-success': models,
                  'bi-x-lg': !models,
                  'text-danger': !models,
                }"
              ></i>
            </div>
          </div>
          <div class="m-3 alert alert-danger" v-if="modelsAlert">
            {{ modelsAlert }}
          </div>
          <div class="m-3" v-if="models">
            <label class="form-label" for="model-input">
              Model
            </label>
            <select class="form-select" id="model-input" v-model="modelId">
              <option v-for="model in models" :key="model.id">
                {{ model.id }}
              </option>
            </select>
          </div>
          <div class="m-3">
            <label class="form-label" for="stop-input">
              Stop sequence
            </label>
            <input
              type="text"
              class="form-control"
              id="stop-input"
              v-model="stop"
            />
          </div>
          <div class="m-3">
            <label class="form-label" for="max-new-tokens-input">
              Max. new tokens
            </label>
            <input
              type="text"
              pattern="[1-9][0-9]*"
              class="form-control"
              id="max-new-tokens-input"
              v-model.number="maxNewTokens"
            />
          </div>
          <div class="m-3">
            <input
              type="checkbox"
              class="form-check-input me-1"
              id="strip-trailing-whitespace-input"
              v-model="stripTrailingWhitespace"
            />
            <label class="form-check-label" for="strip-trailing-whitespace-input">
              Strip trailing whitespace
            </label>
          </div>
          <div class="m-3">
            <label class="form-label" for="completion-suffix-json-input">
              Completion suffix
            </label>
            <input
              type="text"
              class="form-control"
              id="completion-suffix-json-input"
              v-model="completionSuffixJSONString"
            />
          </div>
        </div>
      </div>
    </main>
    <footer class="my-5">
      <div class="row">
        <div class="col-4">
          <div class="card">
            <div class="card-body">
              <h6 class="card-title">Issues?</h6>
              <p class="card-text">
                If something doesn't work, or if something could work better,
                please let us know!
              </p>
              <a
                href="https://github.com/hltcoe/openaisle/issues/new"
                class="card-link"
                >Create issue on GitHub</a
              >
            </div>
          </div>
        </div>
        <div class="col-4">
          <div class="card">
            <div class="card-body">
              <h6 class="card-title">API Documentation</h6>
              <p class="card-text">
                A subset of the OpenAI API is currently implemented on top of
                HuggingFace.
              </p>
              <a
                href="https://hltcoe.github.io/openaisle/"
                class="card-link"
                >Read API docs</a
              >
            </div>
          </div>
        </div>
        <div class="col-4">
          <div class="card">
            <div class="card-body">
              <h6 class="card-title">Source Code</h6>
              <p class="card-text">Want to use or modify this code?</p>
              <a href="https://github.com/hltcoe/openaisle" class="card-link"
                >Go to GitHub repository</a
              >
            </div>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import axios from "axios";

const BASE_64_REGEX = /^[A-Za-z0-9+/=]*$/;

function formatAxiosError(e) {
  if (e.response) {
    if (e.response.status === 400) {
      return "Bad request.";
    } else if (e.response.status === 401) {
      return "Invalid authorization.";
    } else if (e.response.status === 403) {
      return "Authorization is not sufficient.";
    } else if (e.response.status === 504) {
      return "Request timed out.";
    } else if (e.response.status > 0) {
      return `HTTP ${e.response.status}: ${e.response.data}`;
    } else {
      return `${e}`;
    }
  } else if (e.request) {
    return "Received no response from server.";
  } else {
    return e.message;
  }
}

export default {
  data() {
    return {
      apiKey: null,
      modelId: "facebook/opt-2.7b",
      models: null,
      modelsAlert: null,
      stop: "Q:",
      maxNewTokens: 20,
      text: "",
      completionsAlert: null,
      runningCompletions: false,
      stripTrailingWhitespace: true,
      completionSuffixJSONString: "\\n\\nQ: ",
    };
  },
  computed: {
    safeAPIKey() {
      return this.apiKey && BASE_64_REGEX.test(this.apiKey) ? this.apiKey : "";
    },
    openAisleHeaders() {
      return {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.safeAPIKey}`,
      };
    },
    completionSuffix() {
      return JSON.parse('"' + this.completionSuffixJSONString + '"');
    },
  },
  methods: {
    async populateModels() {
      const previousModelId = this.modelId;
      this.models = null;
      this.modelId = null;
      this.models = await this.getModels();
      if (this.models && this.models.length !== 0) {
        if (this.models.find((m) => m.id === previousModelId)) {
          this.modelId = previousModelId;
        } else {
          this.modelId = this.models[0].id;
        }
      } else {
        this.modelId = previousModelId;
      }
    },
    async onSubmit(text) {
      this.text = text;
      const completions = await this.getCompletion(text);
      if (completions !== null) {
        const generatedText = completions.choices[0].text;
        if (generatedText !== null) {
          this.text =
            (this.stripTrailingWhitespace
              ? generatedText.trimEnd()
              : generatedText) + this.completionSuffix;
        }
      }
    },
    async getModels() {
      this.modelsAlert = null;
      try {
        const url = `http://${import.meta.env.VITE_OPENAISLE_HOST}:${
          import.meta.env.VITE_OPENAISLE_PORT
        }/v1/models`;
        const response = await axios.get(url, {
          headers: this.openAisleHeaders,
        });
        return response.data.data;
      } catch (e) {
        console.log(e);
        this.modelsAlert = formatAxiosError(e);
        return null;
      }
    },
    async getCompletion(text) {
      this.completionsAlert = null;
      try {
        const url = `http://${import.meta.env.VITE_OPENAISLE_HOST}:${
          import.meta.env.VITE_OPENAISLE_PORT
        }/v1/completions`;
        this.runningCompletions = true;
        const response = await axios.post(
          url,
          {
            model: this.modelId,
            prompt: text,
            max_tokens: this.maxNewTokens,
            stop: this.stop ? this.stop : null,
          },
          { headers: this.openAisleHeaders }
        );
        return response.data;
      } catch (e) {
        console.log(e);
        this.completionsAlert = formatAxiosError(e);
        return null;
      } finally {
        this.runningCompletions = false;
      }
    },
  },
  watch: {
    apiKey: {
      handler(newValue) {
        this.populateModels();
      },
      immediate: true,
    },
    text(newValue) {
      this.myText = newValue;
    },
  },
};
</script>

<style src="bootstrap/dist/css/bootstrap.min.css">
</style>

<style src="bootstrap-icons/font/bootstrap-icons.css">
</style>

<style src="github-fork-ribbon-css/gh-fork-ribbon.css">
</style>

<style scoped>
.github-fork-ribbon:before {
  background-color: #000;
}
</style>

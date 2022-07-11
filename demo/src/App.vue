
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
          <form class="my-3" @submit.prevent="getCompletionsAndUpdateText">
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
                ref="textbox"
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
            <label class="form-label" for="api-key-input"> API key </label>
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
            <label class="form-label" for="model-input"> Model </label>
            <select class="form-select" id="model-input" v-model="modelId">
              <option v-for="model in models" :key="model.id">
                {{ model.id }}
              </option>
            </select>
          </div>
          <div class="m-3">
            <label class="form-label" for="stop-json-input">
              Stop sequence
            </label>
            <input
              type="text"
              class="form-control"
              id="stop-json-input"
              v-model="stopJSONString"
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
            <label
              class="form-check-label"
              for="strip-trailing-whitespace-input"
            >
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
              <a href="https://hltcoe.github.io/openaisle/" class="card-link"
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
import { nextTick } from "vue";
import axios from "axios";
import { SSE } from "sse.js";

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
      stopJSONString: "\\n",
      maxNewTokens: 20,
      text: `I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q: Where is the Valley of Kings?
A:`,
      completionsAlert: null,
      runningCompletions: false,
      stripTrailingWhitespace: true,
      completionSuffixJSONString: "\\n\\nQ:",
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
    stop() {
      return JSON.parse('"' + this.stopJSONString + '"');
    },
  },
  methods: {
    async populateModels() {
      const previousModelId = this.modelId;
      this.models = null;
      this.modelId = null;
      this.models = await this.getModels();
      if (this.models && this.models.length > 0) {
        if (this.models.find((m) => m.id === previousModelId)) {
          this.modelId = previousModelId;
        } else {
          this.modelId = this.models[0].id;
        }
      } else {
        this.modelId = previousModelId;
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
    handleCompletionsMessage(event) {
      if (event.data !== "[DONE]") {
        const completions = JSON.parse(event.data);
        if (completions !== null) {
          const generatedText = completions.choices[0].text;
          if (generatedText !== null) {
            this.text += generatedText;
          }
        }
      } else {
        this.text =
          (this.stripTrailingWhitespace ? this.text.trimEnd() : this.text) +
          this.completionSuffix;
        this.runningCompletions = false;
      }
      nextTick(() => (this.$refs.textbox.scrollTop = this.$refs.textbox.scrollHeight));
    },
    handleCompletionsError(event) {
      console.log(event);
      this.completionsAlert = `${event.type}: ${event.detail}`;
      this.runningCompletions = false;
    },
    async getCompletionsAndUpdateText() {
      this.completionsAlert = null;
      try {
        const url = `http://${import.meta.env.VITE_OPENAISLE_HOST}:${
          import.meta.env.VITE_OPENAISLE_PORT
        }/v1/completions`;
        const payload = JSON.stringify({
          model: this.modelId,
          prompt: this.text,
          max_tokens: this.maxNewTokens,
          stop: this.stop ? this.stop : null,
          stream: true,
        });
        this.runningCompletions = true;
        const source = new SSE(url, {
          payload: payload,
          headers: this.openAisleHeaders,
        });
        source.addEventListener("message", this.handleCompletionsMessage);
        source.addEventListener("error", this.handleCompletionsError);
        source.addEventListener("abort", this.handleCompletionsError);
        source.stream();
      } catch (e) {
        console.log(e);
        this.completionsAlert = `${e}`;
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

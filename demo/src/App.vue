
<template>
  <div class="container my-5">
    <header class="my-4">
      <h1>SANDLE Demo &#x2B50;</h1>
      <h6 class="text-muted">
        This web app uses a subset of the OpenAI API to communicate with a local
        deployment of large language models in HuggingFace.
      </h6>
    </header>

    <main>
      <div class="row">
        <div class="col-8">
          <form class="my-3" @submit.prevent="getCompletionsAndUpdatePrompt">
            <div class="my-3">
              <label class="form-label"
                >Enter some text, then click "Submit" to generate a
                completion.</label
              >
              <textarea
                class="form-control"
                :disabled="runningCompletions"
                rows="20"
                placeholder="Say this is a test."
                v-model="prompt"
                ref="textbox"
                @keypress.ctrl.enter.prevent="getCompletionsAndUpdatePrompt"
              />
            </div>
            <div class="my-3 alert alert-danger" v-if="completionsError">
              {{ completionsError }}
            </div>
            <div class="my-3 alert alert-warning" v-if="completionsWarning">
              {{ completionsWarning }}
            </div>
            <div class="my-3 d-flex">
              <div>
                <button
                  type="submit"
                  class="btn btn-primary"
                  :class="{ disabled: runningCompletions }"
                >
                  Submit
                </button>
                <p class="form-text">Ctrl + Enter</p>
              </div>
              <div class="ms-2">
                <button
                  type="button"
                  class="btn btn-secondary"
                  :class="{ disabled: runningCompletions || useGreedyDecoding }"
                  title="Resample previous completion"
                  @click="redoPreviousCompletions"
                >
                  <i class="bi bi-arrow-clockwise text-bold"></i> Redo previous
                </button>
              </div>
              <div
                class="spinner-border ms-3"
                role="status"
                v-if="runningCompletions"
              >
                <span class="visually-hidden">Loading...</span>
              </div>
            </div>
          </form>
        </div>
        <div class="col-4">
          <div class="m-3">
            <label class="form-label" for="api-key-input">
              API key or password
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
                  'bi-check-lg': !!models,
                  'text-success': !!models,
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
              <option v-for="model in models" :value="model.id" :key="model.id">
                {{ model.description }}
              </option>
            </select>
          </div>
          <div class="m-3">
            <label class="form-label" for="stop-json-input">
              Stop sequence
              <a
                @click.prevent
                href="#"
                data-bs-toggle="tooltip"
                title="Truncate text generated after (and including) this string.  This string is JSON-encoded, so newline is \n, tab is \t, etc."
                ><i class="bi bi-question-circle text-muted"></i
              ></a>
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
              <a
                @click.prevent
                href="#"
                data-bs-toggle="tooltip"
                title="Maximum number of new tokens to generate."
                ><i class="bi bi-question-circle text-muted"></i
              ></a>
            </label>
            <input
              type="number"
              class="form-control"
              id="max-new-tokens-input"
              v-model.number="maxNewTokens"
            />
          </div>
          <div class="m-3">
            <input
              type="checkbox"
              class="form-check-input me-2"
              id="use-greedy-decoding-input"
              v-model="useGreedyDecoding"
            />
            <label class="form-check-label" for="use-greedy-decoding-input">
              Use greedy decoding
              <a
                @click.prevent
                href="#"
                data-bs-toggle="tooltip"
                title="If unchecked, use random sampling."
                ><i class="bi bi-question-circle text-muted"></i
              ></a>
            </label>
          </div>
          <div class="m-3">
            <label
              class="form-label d-flex justify-content-between"
              for="temperature-input"
            >
              <span>
                Temperature
                <a
                  @click.prevent
                  href="#"
                  data-bs-toggle="tooltip"
                  title="Softmax temperature to sample at (higher is more random)."
                  ><i class="bi bi-question-circle text-muted"></i
                ></a>
              </span>
              <span class="text-muted mx-2">{{ temperature }}</span>
            </label>
            <input
              type="range"
              class="form-range"
              id="temperature-input"
              min="0"
              max="1"
              step="0.01"
              v-model.number="temperature"
            />
          </div>
          <div class="m-3">
            <label
              class="form-label d-flex justify-content-between"
              for="top-p-input"
            >
              <span>
                Top-p
                <a
                  @click.prevent
                  href="#"
                  data-bs-toggle="tooltip"
                  title="Probability mass to sample from (higher is more random)."
                  ><i class="bi bi-question-circle text-muted"></i
                ></a>
              </span>
              <span class="text-muted mx-2">{{ topP }}</span>
            </label>
            <input
              type="range"
              class="form-range"
              id="top-p-input"
              min="0"
              max="1"
              step="0.01"
              :disabled="useGreedyDecoding"
              v-model.number="topP"
            />
          </div>
          <div class="m-3">
            <input
              type="checkbox"
              class="form-check-input me-2"
              id="strip-trailing-whitespace-input"
              v-model="stripTrailingWhitespace"
            />
            <label
              class="form-check-label"
              for="strip-trailing-whitespace-input"
            >
              Strip trailing whitespace
              <a
                @click.prevent
                href="#"
                data-bs-toggle="tooltip"
                title="If checked, strip trailing whitespace from generated text."
                ><i class="bi bi-question-circle text-muted"></i
              ></a>
            </label>
          </div>
          <div class="m-3">
            <label class="form-label" for="completion-suffix-json-input">
              Completion suffix
              <a
                @click.prevent
                href="#"
                data-bs-toggle="tooltip"
                title="Append this string to the end of generated text.  This string is JSON-encoded, so newline is \n, tab is \t, etc."
                ><i class="bi bi-question-circle text-muted"></i
              ></a>
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
                href="https://github.com/hltcoe/sandle/issues/new"
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
              <a href="https://hltcoe.github.io/sandle/" class="card-link"
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
              <a href="https://github.com/hltcoe/sandle" class="card-link"
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
import "@popperjs/core";
import { Tooltip } from "bootstrap";
import { nextTick } from "vue";
import axios from "axios";
import { SSE } from "sse.js";
import * as Sentry from "@sentry/vue";

const DEFAULT_TEMPERATURE = 0.7;
const BASE_64_REGEX = /^[A-Za-z0-9+/=]*$/;


function formatAxiosError(e) {
  if (
    e.response &&
    e.response.data &&
    e.response.data.error &&
    e.response.data.error.message
  ) {
    return e.response.data.error.message;
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
      temperature: DEFAULT_TEMPERATURE,
      previousTemperature: null,
      topP: 1,
      prompt: `I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

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
      completionsError: null,
      completionsWarning: null,
      runningCompletions: false,
      previousCompletionsPrompt: null,
      stripTrailingWhitespace: true,
      useGreedyDecoding: false,
      completionSuffixJSONString: "\\n\\nQ:",
      tooltips: null,
    };
  },
  computed: {
    safeAPIKey() {
      return this.apiKey && BASE_64_REGEX.test(this.apiKey) ? this.apiKey : "";
    },
    sandleHeaders() {
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
      this.modelsAlert = null;
      try {
        this.models = await this.getModels();
        if (this.models && this.models.length > 0) {
          if (this.models.find((m) => m.id === previousModelId)) {
            this.modelId = previousModelId;
          } else {
            this.modelId = this.models[0].id;
          }
        }
      } catch (e) {
        this.modelId = previousModelId;
        this.modelsAlert = formatAxiosError(e);
        if (!(e.response && [401, 403].includes(e.response.status))) {
          throw e;
        }
      }
    },
    async getModels() {
      const url = `${import.meta.env.VITE_SANDLE_URL_PREFIX}/v1/models`;
      const response = await axios.get(url, {
        headers: this.sandleHeaders,
      });
      return response.data.data;
    },
    handleCompletionsMessage(event) {
      if (event.data !== "[DONE]") {
        const completions = JSON.parse(event.data);
        if (completions !== null) {
          const generatedText = completions.choices[0].text;
          if (generatedText !== null) {
            this.prompt += generatedText;
          }
        }
      } else {
        this.prompt =
          (this.stripTrailingWhitespace ? this.prompt.trimEnd() : this.prompt) +
          this.completionSuffix;
        this.runningCompletions = false;
      }
      nextTick(
        () => (this.$refs.textbox.scrollTop = this.$refs.textbox.scrollHeight)
      );
    },
    handleCompletionsError(event) {
      const message = `Error streaming completions: The backend may be busy or down`;
      this.completionsError = message;
      this.runningCompletions = false;
      throw new Error(message);
    },
    async redoPreviousCompletions() {
      if (this.previousCompletionsPrompt !== null) {
        this.prompt = this.previousCompletionsPrompt;
        await this.getCompletionsAndUpdatePrompt();
      }
    },
    async getCompletionsAndUpdatePrompt() {
      if (!this.runningCompletions) {
        this.completionsError = null;
        this.completionsWarning = null;
        this.previousCompletionsPrompt = this.prompt;
        if (this.prompt.endsWith(" ")) {
          this.completionsWarning =
            "Warning: The prompt ends with a space character which may cause performance issues.";
        }
        try {
          const url = `${
            import.meta.env.VITE_SANDLE_URL_PREFIX
          }/v1/completions`;
          const payload = JSON.stringify({
            model: this.modelId,
            prompt: this.prompt,
            greedy_decoding:
              this.temperature > 0 ? this.useGreedyDecoding : true,
            max_tokens: this.maxNewTokens,
            temperature: this.temperature > 0 ? this.temperature : 1,
            top_p: this.topP,
            stop: this.stop ? this.stop : null,
            stream: true,
          });
          this.runningCompletions = true;
          const source = new SSE(url, {
            payload: payload,
            headers: this.sandleHeaders,
          });
          source.addEventListener("message", this.handleCompletionsMessage);
          source.addEventListener("error", this.handleCompletionsError);
          source.addEventListener("abort", this.handleCompletionsError);
          source.stream();
        } catch (e) {
          this.completionsError = `Error setting up completions stream: ${e}`;
          this.runningCompletions = false;
          throw e;
        }
      }
    },
  },
  mounted() {
    const tooltipTriggers = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    this.tooltips = tooltipTriggers.map(function (tooltipTriggerEl) {
      return new Tooltip(tooltipTriggerEl);
    });
  },
  watch: {
    apiKey: {
      handler(newValue) {
        Sentry.setUser(newValue ? { id: newValue } : null);
        this.populateModels();
      },
      immediate: true,
    },
    temperature: {
      handler(newValue) {
        this.useGreedyDecoding = newValue === 0;
      },
      immediate: true,
    },
    useGreedyDecoding: {
      handler(newValue) {
        if (newValue) {
          this.previousTemperature = this.temperature;
          this.temperature = 0;
        } else {
          this.temperature = this.previousTemperature
            ? this.previousTemperature
            : DEFAULT_TEMPERATURE;
        }
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

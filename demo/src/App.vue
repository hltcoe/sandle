<script>
import PromptForm from './components/PromptForm.vue'
import axios from 'axios'

function formatAxiosError(e) {
  if (e.response) {
    if (e.response.status === 400) {
      return 'Bad request.';
    } if (e.response.status === 401) {
      return 'Invalid authorization.';
    } else if (e.response.status === 403) {
      return 'Authorization is not sufficient.';
    } else {
      return `HTTP ${e.response.status}: ${e.response.data}`;
    }
  } else if (e.request) {
    return 'Received no response from server.';
  } else {
    return e.message;
  }
}

export default {
  components: {
    PromptForm,
  },
  data() {
    return {
      text: '',
      models: null,
      modelsAlert: null,
    };
  },
  methods: {
    async onChange(text) {
      this.text = text;
      const completions = await postCompletions(text);
      const generatedText = completions.choices[0].text;
      if (generatedText !== null) {
        this.text = generatedText;
      }
    },
    async getModels() {
      this.modelsAlert = null;
      try {
        const url = `http://${process.env.VUE_APP_OPENAISLE_HOST}:${process.env.VUE_APP_OPENAISLE_PORT}/v1/models`;
        const response = await axios.get(
          url,
          { headers: { "Content-Type": "application/json" } }
        );
        return response.data;
      } catch (e) {
        console.log(e);
        this.modelsAlert = formatAxiosError(e);
        return null;
      }
    },
    async postCompletions(text) {
      this.completionsAlert = null;
      try {
        const url = `http://${process.env.VUE_APP_OPENAISLE_HOST}:${process.env.VUE_APP_OPENAISLE_PORT}/v1/completions`;
        const response = await axios.post(
          url,
          { model: 'facebook/opt-125m', prompt: text },
          { headers: { "Content-Type": "application/json" } }
        );
        return response.data;
      } catch (e) {
        console.log(e);
        this.completionsAlert = formatAxiosError(e);
        return null;
      }
    },
  },
  mounted() {
    (async () => (this.models = this.getModels()))();
  },
}
</script>

<template>
  <div class="container">
    <header>
      <h1>
        You did it! &#x1F495;
      </h1>
    </header>
  
    <main>
      <div class="row">
        <div class="col-8">
          <PromptForm :text="text" @change="onChange" />
        </div>
        <div class="col-4">
          <div class="alert alert-danger" v-if="completionsAlert">
            {{ completionsAlert }}
          </div>
          <div v-else>
            <ul>
              <li v-for="model in models" :key="model.id">{{ model.id }}</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style src="bootstrap/dist/css/bootstrap.min.css">
</style>

<style src="bootstrap-icons/font/bootstrap-icons.css">
</style>

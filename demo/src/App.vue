<script>
import PromptForm from './components/PromptForm.vue'

export default {
  components: {
    PromptForm,
  },
  data() {
    return {
      text: '',
    };
  },
  methods: {
    onChange(text) {
      this.text = text;
    }
  }
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
      <PromptForm :text="text" @change="onChange" />
    </main>
  </div>
</template>

<style src="bootstrap/dist/css/bootstrap.min.css">
</style>

<style src="bootstrap-icons/font/bootstrap-icons.css">
</style>

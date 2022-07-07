<script>
export default {
  props: {
    text: { required: true, type: String },
  },
  emits: ['change'],
  data() {
    return {
      myText: '',
    };
  },
  watch: {
    text(newValue, oldValue) {
      this.myText = newValue;
    },
  },
}
</script>

<template>
  <form @submit.prevent="$emit('change', myText)">
    <div class="my-3">
      <label class="form-label">Text to complete</label>
      <textarea class="form-control" v-model="myText" />
    </div>
    <div class="my-3">
      <button type="submit" class="btn btn-primary">Submit</button>
    </div>
  </form>
</template>

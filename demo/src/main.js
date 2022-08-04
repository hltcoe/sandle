import { createApp } from "vue";
import * as Sentry from "@sentry/vue";
import { BrowserTracing } from "@sentry/tracing";
import { CaptureConsole } from '@sentry/integrations';
import App from './App.vue'

const app = createApp(App);

Sentry.init({
  app,
  dsn: import.meta.env.VITE_SENTRY_DSN,
  release: import.meta.env.VITE_SENTRY_RELEASE,
  environment: import.meta.env.MODE,
  integrations: [
    new BrowserTracing({}),
    new CaptureConsole({levels: ['error']}),
  ],
  tracesSampleRate: 1.0,  // a rate less than 1.0 is recommended in production, yolo
  logErrors: true,
});
Sentry.setTag("component", "demo");

app.mount("#app");

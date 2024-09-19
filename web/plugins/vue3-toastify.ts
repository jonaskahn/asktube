import Vue3Toastify, { toast } from "vue3-toastify";
import "vue3-toastify/dist/index.css";

export default defineNuxtPlugin((nuxtApp) => {
  nuxtApp.vueApp.use(Vue3Toastify, {
    position: toast.POSITION.TOP_CENTER,
    autoClose: 2000,
    toastStyle: {
      "z-index": 1001,
    },
  });

  return {
    provide: { toast },
  };
});

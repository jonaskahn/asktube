import daisyui from "daisyui";
import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: [],
  theme: {
    extend: {},
  },
  plugins: [daisyui, typography],
  daisyui: {
    themes: ["light", "dark", "lofi"],
    darkTheme: "lofi",
  },
};

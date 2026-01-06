/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        phoenix: {
          dark: "#0f172a",
          primary: "#f97316",
          secondary: "#3b82f6",
        }
      }
    },
  },
  plugins: [],
}

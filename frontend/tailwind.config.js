/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#0F2942",
          50: "#EEF3F7",
          100: "#D6E1E9",
          400: "#4A6B85",
          600: "#1D4258",
          800: "#122F45",
          900: "#0B1F30",
        },
        steel: {
          DEFAULT: "#1D4E89",
          500: "#1D4E89",
          600: "#183F6E",
        },
        risk: {
          DEFAULT: "#C1432D",
          50: "#FBEAE6",
          100: "#F4CFC5",
          500: "#C1432D",
          600: "#9E3624",
        },
        safe: {
          DEFAULT: "#2F6F4E",
          50: "#E8F1EC",
          500: "#2F6F4E",
        },
        surface: {
          DEFAULT: "#F7F8FA",
          panel: "#FFFFFF",
          border: "#E2E6EA",
        },
        muted: "#5B6B7A",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Space Grotesk", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

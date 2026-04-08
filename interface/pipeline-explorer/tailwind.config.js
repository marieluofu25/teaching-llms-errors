/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#f7f9fb',
        background: '#f7f9fb',
        'surface-container-low': '#f2f4f6',
        'surface-container-lowest': '#ffffff',
        'surface-container-high': '#e6e8ea',
        'surface-container-highest': '#e0e3e5',
        'on-surface': '#191c1e',
        'on-surface-variant': '#424754',
        'outline-variant': '#c2c6d6',
        outline: '#727785',
        primary: '#0058be',
        'primary-container': '#2170e4',
        secondary: '#4648d4',
        error: '#ba1a1a',
      },
      fontFamily: {
        headline: ['Manrope', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        lg: '0.5rem',
        xl: '0.5rem',
      },
      boxShadow: {
        float: '0 20px 40px rgba(25, 28, 30, 0.06)',
      },
    },
  },
  plugins: [],
}

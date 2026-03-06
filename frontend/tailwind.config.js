/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Geist"', '"DM Sans"', 'ui-sans-serif', 'system-ui'],
        mono: ['"Geist Mono"', '"JetBrains Mono"', 'ui-monospace'],
      },
      colors: {
        surface: {
          50:  '#f7f7f8',
          100: '#ededef',
          200: '#d9d9de',
          800: '#1a1a20',
          900: '#111116',
          950: '#0a0a0d',
        },
        accent: {
          DEFAULT: '#6c63ff',
          hover:   '#5a52e8',
          muted:   '#6c63ff22',
        },
      },
      animation: {
        'fade-up':    'fadeUp 0.3s ease forwards',
        'pulse-dot':  'pulseDot 1.2s ease-in-out infinite',
        'slide-in':   'slideIn 0.25s ease forwards',
      },
      keyframes: {
        fadeUp:   { from: { opacity: 0, transform: 'translateY(8px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        pulseDot: { '0%,100%': { opacity: 0.3, transform: 'scale(0.8)' }, '50%': { opacity: 1, transform: 'scale(1.2)' } },
        slideIn:  { from: { opacity: 0, transform: 'translateX(-8px)' }, to: { opacity: 1, transform: 'translateX(0)' } },
      },
    },
  },
  plugins: [],
}

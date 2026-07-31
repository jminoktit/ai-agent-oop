import { useState, useEffect } from 'react';

export default function ThemeSwitcher() {
  const [theme, setTheme] = useState(() => localStorage.getItem('aura_theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('aura_theme', theme);

    if (theme === 'light') {
      document.documentElement.style.setProperty('--bg-primary', '#f8f9fc');
      document.documentElement.style.setProperty('--bg-secondary', '#ffffff');
      document.documentElement.style.setProperty('--bg-tertiary', '#f0f1f5');
      document.documentElement.style.setProperty('--bg-hover', '#e8e9ee');
      document.documentElement.style.setProperty('--bg-active', '#dddee5');
      document.documentElement.style.setProperty('--border', '#dddee5');
      document.documentElement.style.setProperty('--border-light', '#c8c9d0');
      document.documentElement.style.setProperty('--text-primary', '#1a1a2e');
      document.documentElement.style.setProperty('--text-secondary', '#555570');
      document.documentElement.style.setProperty('--text-muted', '#8888a0');
    } else {
      document.documentElement.style.setProperty('--bg-primary', '#0a0a0f');
      document.documentElement.style.setProperty('--bg-secondary', '#111118');
      document.documentElement.style.setProperty('--bg-tertiary', '#1a1a24');
      document.documentElement.style.setProperty('--bg-hover', '#222230');
      document.documentElement.style.setProperty('--bg-active', '#2a2a3a');
      document.documentElement.style.setProperty('--border', '#2a2a3a');
      document.documentElement.style.setProperty('--border-light', '#333345');
      document.documentElement.style.setProperty('--text-primary', '#f0f0f5');
      document.documentElement.style.setProperty('--text-secondary', '#a0a0b5');
      document.documentElement.style.setProperty('--text-muted', '#606075');
    }
  }, [theme]);

  return (
    <button
      className="theme-switcher"
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}

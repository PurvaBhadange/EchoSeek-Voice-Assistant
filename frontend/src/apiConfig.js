/**
 * API Base URL Configuration for EchoSeek Frontend
 * Direct connection to live Render backend eliminates Vercel proxy 502 gateway timeouts.
 */
export const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || 
  (window.location.hostname === 'localhost' ? '' : 'https://echoseek-voice-assistant.onrender.com');

export const getApiUrl = (path) => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};

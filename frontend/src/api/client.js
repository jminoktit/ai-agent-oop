const API_BASE = '';

function getCookie(name) {
  const cookies = document.cookie.split(';');
  for (const c of cookies) {
    const [key, val] = c.trim().split('=');
    if (key === name) return val;
  }
  return null;
}

const CSRF_TOKEN = () => getCookie('csrftoken');

async function request(url, options = {}) {
  const headers = { ...options.headers };
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';
  }
  if (!['GET', 'HEAD', 'OPTIONS'].includes(options.method?.toUpperCase() || 'GET')) {
    headers['X-CSRFToken'] = CSRF_TOKEN();
  }

  const config = { ...options, headers };

  if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
    config.body = JSON.stringify(config.body);
  }

  const res = await fetch(`${API_BASE}${url}`, config);
  const contentType = res.headers.get('content-type') || '';
  let data;
  if (contentType.includes('application/json')) {
    data = await res.json();
  } else {
    data = await res.text();
  }

  if (!res.ok) {
    const msg = typeof data === 'object' ? (data.error || data.message || 'Request failed') : 'Request failed';
    throw new Error(msg);
  }

  return data;
}

export const api = {
  // Auth
  login: (username, password) =>
    request('/login/', {
      method: 'POST',
      body: { username, password },
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    }),
  register: (username, password1, password2) =>
    request('/register/', {
      method: 'POST',
      body: { username, password1, password2 },
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    }),
  logout: () => request('/logout/'),

  // Chat
  chat: (message, conversationId = null, agent = null) =>
    request('/chat/', { method: 'POST', body: { message, conversation_id: conversationId, agent } }),
  switchAgent: (agent) =>
    request('/switch-agent/', { method: 'POST', body: { agent } }),
  newConversation: (agent) =>
    request('/new-conversation/', { method: 'POST', body: { agent } }),

  // Conversations
  conversations: () => request('/conversations/'),
  getConversation: (id) => request(`/conversation/${id}/`, { headers: { Accept: 'application/json' } }),
  renameConversation: (id, title) =>
    request(`/conversation/${id}/rename/`, { method: 'POST', body: { title } }),
  deleteConversation: (id) =>
    request(`/conversation/${id}/delete/`, { method: 'POST' }),
  clearConversation: (id) =>
    request(`/conversation/${id}/clear/`, { method: 'POST' }),
  togglePin: (id) =>
    request(`/conversation/${id}/pin/`, { method: 'POST' }),

  // Agents
  agentInfo: () => request('/agent-info/'),
  userInfo: () => request('/user-info/'),

  // Files
  listFiles: () => request('/files/'),
  uploadFile: async (formData) => {
    const res = await fetch(`${API_BASE}/files/upload/`, {
      method: 'POST',
      headers: { 'X-CSRFToken': CSRF_TOKEN() },
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    return data;
  },
  getFileContent: (id) => request(`/files/${id}/`),
  deleteFile: (id) => request(`/files/${id}/delete/`, { method: 'POST' }),

  // Training
  trainingDashboard: () => request('/training/', { headers: { Accept: 'application/json' } }),
  startTraining: (data) =>
    request('/training/start/', { method: 'POST', body: data }),
  trainingStatus: (jobId) =>
    request(`/training/${jobId}/status/`),
  stopTraining: (jobId) =>
    request(`/training/${jobId}/stop/`, { method: 'POST' }),

  // Settings
  getSettings: () => request('/settings/get/'),
  updateProfile: (data) =>
    request('/settings/profile/', { method: 'POST', body: data }),
  updateChatSettings: (data) =>
    request('/settings/chat/', { method: 'POST', body: data }),
  updateTrainingSettings: (data) =>
    request('/settings/training/', { method: 'POST', body: data }),
  updateApiKeys: (data) =>
    request('/settings/api/', { method: 'POST', body: data }),
};

import { useState, useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useLang } from '../i18n/LanguageContext';
import { api } from '../api/client';

export default function SettingsPage() {
  const { t } = useLang();
  const { tab: urlTab } = useParams();
  const [activeTab, setActiveTab] = useState(urlTab || 'profile');
  const [toast, setToast] = useState(null);
  const [settings, setSettings] = useState({
    display_name: '',
    avatar_url: '',
    theme: 'dark',
    language: 'en',
    default_agent: 'chat',
    show_timestamps: true,
    send_on_enter: true,
    font_size: 14,
    email_notifications: true,
    training_notifications: true,
    sound_enabled: true,
    default_model: 'google/gemma-2-2b-it',
    default_dataset_size: '100k',
    default_epochs: 3,
    default_learning_rate: 2e-4,
    openai_api_key: '',
    huggingface_token: '',
    smtp_user: '',
    smtp_pass: '',
  });

  useEffect(() => {
    loadSettings();
  }, []);

  useEffect(() => {
    if (urlTab) setActiveTab(urlTab);
  }, [urlTab]);

  const loadSettings = async () => {
    try {
      const data = await api.getSettings();
      setSettings((prev) => ({ ...prev, ...data }));
    } catch {
      // Use defaults
    }
  };

  const handleSave = async (section) => {
    try {
      switch (section) {
        case 'profile':
          await api.updateProfile({
            display_name: settings.display_name,
            avatar_url: settings.avatar_url,
            theme: settings.theme,
            language: settings.language,
          });
          break;
        case 'chat':
          await api.updateChatSettings({
            default_agent: settings.default_agent,
            show_timestamps: settings.show_timestamps,
            send_on_enter: settings.send_on_enter,
            font_size: settings.font_size,
          });
          break;
        case 'training':
          await api.updateTrainingSettings({
            default_model: settings.default_model,
            default_dataset_size: settings.default_dataset_size,
            default_epochs: settings.default_epochs,
            default_learning_rate: settings.default_learning_rate,
          });
          break;
        case 'api':
          await api.updateApiKeys({
            openai_api_key: settings.openai_api_key,
            huggingface_token: settings.huggingface_token,
            smtp_user: settings.smtp_user,
            smtp_pass: settings.smtp_pass,
          });
          break;
        case 'notifications':
          // Notifications saved through profile
          await api.updateProfile({
            email_notifications: settings.email_notifications,
            training_notifications: settings.training_notifications,
            sound_enabled: settings.sound_enabled,
          });
          break;
      }
      showToast(t('settingsSaved'), 'success');
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const tabs = [
    { id: 'profile', icon: '👤', label: t('profile') },
    { id: 'chat', icon: '💬', label: t('chatSettings') },
    { id: 'training', icon: '🧠', label: t('trainingSettings') },
    { id: 'notifications', icon: '🔔', label: t('notificationSettings') },
    { id: 'api', icon: '🔑', label: t('apiKeys') },
  ];

  const update = (key, value) => setSettings((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="settings-page">
      {/* Header */}
      <div className="settings-header">
        <Link to="/" style={{ color: 'var(--text-secondary)', textDecoration: 'none', fontSize: '14px' }}>
          ← {t('chat')}
        </Link>
        <h1>{t('settingsPage')}</h1>
      </div>

      <div className="settings-content">
        {/* Sidebar */}
        <div className="settings-sidebar">
          <nav className="settings-nav">
            {tabs.map((tab) => (
              <a
                key={tab.id}
                className={activeTab === tab.id ? 'active' : ''}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.icon} {tab.label}
              </a>
            ))}
          </nav>
        </div>

        {/* Panel */}
        <div className="settings-panel">
          {activeTab === 'profile' && (
            <>
              <h2>{t('profile')}</h2>
              <p>Manage your profile settings</p>

              <div className="settings-group">
                <label>{t('displayName')}</label>
                <input
                  className="settings-input"
                  value={settings.display_name}
                  onChange={(e) => update('display_name', e.target.value)}
                  placeholder="Your name"
                />
              </div>

              <div className="settings-group">
                <label>{t('avatarUrl')}</label>
                <input
                  className="settings-input"
                  value={settings.avatar_url}
                  onChange={(e) => update('avatar_url', e.target.value)}
                  placeholder="https://example.com/avatar.jpg"
                />
              </div>

              <div className="settings-group">
                <label>{t('theme')}</label>
                <select className="settings-select" value={settings.theme} onChange={(e) => update('theme', e.target.value)}>
                  <option value="dark">{t('dark')}</option>
                  <option value="light">{t('light')}</option>
                  <option value="auto">{t('auto')}</option>
                </select>
              </div>

              <div className="settings-group">
                <label>{t('language')}</label>
                <select className="settings-select" value={settings.language} onChange={(e) => update('language', e.target.value)}>
                  <option value="en">{t('english')}</option>
                  <option value="ar">{t('arabic')}</option>
                </select>
              </div>

              <button className="settings-save-btn" onClick={() => handleSave('profile')}>
                {t('save')}
              </button>
            </>
          )}

          {activeTab === 'chat' && (
            <>
              <h2>{t('chatSettings')}</h2>
              <p>Customize your chat experience</p>

              <div className="settings-group">
                <label>{t('defaultAgent')}</label>
                <select className="settings-select" value={settings.default_agent} onChange={(e) => update('default_agent', e.target.value)}>
                  <option value="chat">ChatBot</option>
                  <option value="code">CodeBot</option>
                  <option value="data">DataBot</option>
                  <option value="research">ResearchBot</option>
                  <option value="planner">PlannerBot</option>
                  <option value="media">MediaBot</option>
                </select>
              </div>

              <div className="settings-group">
                <label>{t('fontSize')} ({settings.font_size}px)</label>
                <input
                  type="range"
                  min="12"
                  max="20"
                  value={settings.font_size}
                  onChange={(e) => update('font_size', parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div className="settings-toggle">
                <span style={{ fontSize: '14px' }}>{t('showTimestamps')}</span>
                <div
                  className={`toggle-switch ${settings.show_timestamps ? 'active' : ''}`}
                  onClick={() => update('show_timestamps', !settings.show_timestamps)}
                />
              </div>

              <div className="settings-toggle">
                <span style={{ fontSize: '14px' }}>{t('sendOnEnter')}</span>
                <div
                  className={`toggle-switch ${settings.send_on_enter ? 'active' : ''}`}
                  onClick={() => update('send_on_enter', !settings.send_on_enter)}
                />
              </div>

              <button className="settings-save-btn" onClick={() => handleSave('chat')}>
                {t('save')}
              </button>
            </>
          )}

          {activeTab === 'training' && (
            <>
              <h2>{t('trainingSettings')}</h2>
              <p>Set default values for training jobs</p>

              <div className="settings-group">
                <label>{t('defaultModel')}</label>
                <select className="settings-select" value={settings.default_model} onChange={(e) => update('default_model', e.target.value)}>
                  <option value="google/gemma-2-2b-it">Google Gemma-2-2B-it</option>
                  <option value="google/gemma-2-9b-it">Google Gemma-2-9B-it</option>
                  <option value="meta-llama/Llama-3-8B">Meta Llama-3-8B</option>
                  <option value="Qwen/Qwen2-7B">Qwen2-7B</option>
                </select>
              </div>

              <div className="settings-group">
                <label>{t('defaultDatasetSize')}</label>
                <select className="settings-select" value={settings.default_dataset_size} onChange={(e) => update('default_dataset_size', e.target.value)}>
                  <option value="10k">10k</option>
                  <option value="50k">50k</option>
                  <option value="100k">100k</option>
                  <option value="200k">200k</option>
                  <option value="500k">500k</option>
                </select>
              </div>

              <div className="settings-group">
                <label>{t('defaultEpochs')}</label>
                <input
                  type="number"
                  className="settings-input"
                  value={settings.default_epochs}
                  onChange={(e) => update('default_epochs', parseInt(e.target.value) || 3)}
                  min="1"
                  max="10"
                />
              </div>

              <div className="settings-group">
                <label>{t('defaultLearningRate')}</label>
                <input
                  type="number"
                  className="settings-input"
                  value={settings.default_learning_rate}
                  onChange={(e) => update('default_learning_rate', parseFloat(e.target.value) || 2e-4)}
                  step="0.00001"
                />
              </div>

              <button className="settings-save-btn" onClick={() => handleSave('training')}>
                {t('save')}
              </button>
            </>
          )}

          {activeTab === 'notifications' && (
            <>
              <h2>{t('notificationSettings')}</h2>
              <p>Manage notification preferences</p>

              <div className="settings-toggle">
                <span style={{ fontSize: '14px' }}>{t('emailNotifications')}</span>
                <div
                  className={`toggle-switch ${settings.email_notifications ? 'active' : ''}`}
                  onClick={() => update('email_notifications', !settings.email_notifications)}
                />
              </div>

              <div className="settings-toggle">
                <span style={{ fontSize: '14px' }}>{t('trainingNotifications')}</span>
                <div
                  className={`toggle-switch ${settings.training_notifications ? 'active' : ''}`}
                  onClick={() => update('training_notifications', !settings.training_notifications)}
                />
              </div>

              <div className="settings-toggle">
                <span style={{ fontSize: '14px' }}>{t('soundEnabled')}</span>
                <div
                  className={`toggle-switch ${settings.sound_enabled ? 'active' : ''}`}
                  onClick={() => update('sound_enabled', !settings.sound_enabled)}
                />
              </div>

              <button className="settings-save-btn" onClick={() => handleSave('notifications')}>
                {t('save')}
              </button>
            </>
          )}

          {activeTab === 'api' && (
            <>
              <h2>{t('apiKeys')}</h2>
              <p>Configure API keys and tokens</p>

              <div className="settings-group">
                <label>{t('openaiApiKey')}</label>
                <input
                  type="password"
                  className="settings-input"
                  value={settings.openai_api_key}
                  onChange={(e) => update('openai_api_key', e.target.value)}
                  placeholder="sk-..."
                />
              </div>

              <div className="settings-group">
                <label>{t('huggingfaceToken')}</label>
                <input
                  type="password"
                  className="settings-input"
                  value={settings.huggingface_token}
                  onChange={(e) => update('huggingface_token', e.target.value)}
                  placeholder="hf_..."
                />
              </div>

              <div className="settings-group">
                <label>{t('smtpUser')}</label>
                <input
                  type="text"
                  className="settings-input"
                  value={settings.smtp_user}
                  onChange={(e) => update('smtp_user', e.target.value)}
                  placeholder="your@gmail.com"
                />
              </div>

              <div className="settings-group">
                <label>{t('smtpPass')}</label>
                <input
                  type="password"
                  className="settings-input"
                  value={settings.smtp_pass}
                  onChange={(e) => update('smtp_pass', e.target.value)}
                  placeholder="xxxx-xxxx-xxxx-xxxx"
                />
              </div>

              <button className="settings-save-btn" onClick={() => handleSave('api')}>
                {t('save')}
              </button>
            </>
          )}
        </div>
      </div>

      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}

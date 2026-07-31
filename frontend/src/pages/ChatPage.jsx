import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useLang } from '../i18n/LanguageContext';
import { api } from '../api/client';
import MessageContent from '../components/MessageContent';
import MessageActions from '../components/MessageActions';
import VoiceInput from '../components/VoiceInput';
import CommandPalette from '../components/CommandPalette';
import ThemeSwitcher from '../components/ThemeSwitcher';
import DragDropUpload from '../components/DragDropUpload';
import { exportAsMarkdown, exportAsJson } from '../utils/export';

const AGENT_ICONS = { chat: '💬', code: '💻', data: '📊', research: '🔍', planner: '📋', media: '🎨' };
const AGENT_COLORS = { chat: '#6c5ce7', code: '#00b4d8', data: '#00d68f', research: '#ffaa00', planner: '#ff3d71', media: '#a855f7' };

export default function ChatPage() {
  const { t, lang, setLang } = useLang();
  const navigate = useNavigate();

  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState('');
  const [activeAgent, setActiveAgent] = useState('chat');
  const [agents, setAgents] = useState([]);
  const [userName, setUserName] = useState('');
  const [renameId, setRenameId] = useState(null);
  const [renameTitle, setRenameTitle] = useState('');
  const [cmdOpen, setCmdOpen] = useState(false);
  const [dragFiles, setDragFiles] = useState(false);
  const [toast, setToast] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { loadData(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streaming]);

  // Ctrl+K shortcut
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCmdOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const loadData = async () => {
    try {
      const [convData, agentData, userData] = await Promise.all([
        api.conversations().catch(() => ({ conversations: [] })),
        api.agentInfo().catch(() => ({ agents: ['chat', 'code', 'data', 'research', 'planner', 'media'], active_agent: 'chat' })),
        api.userInfo().catch(() => ({ username: 'User' })),
      ]);
      setConversations(convData.conversations || []);
      setAgents(agentData.agents || []);
      setActiveAgent(agentData.active_agent || 'chat');
      setUserName(userData.username || 'User');
    } catch { setAgents(['chat', 'code', 'data', 'research', 'planner', 'media']); }
  };

  const selectConversation = async (conv) => {
    setActiveConvId(conv.id);
    try {
      const data = await api.getConversation(conv.id);
      setMessages(data.messages || []);
      if (data.agent_key) setActiveAgent(data.agent_key);
    } catch (err) { console.error(err); }
  };

  const handleNewChat = async () => {
    try { await api.newConversation(activeAgent); } catch {}
    setActiveConvId(null);
    setMessages([]);
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setStreaming('');

    try {
      // Try streaming first
      const csrfToken = document.cookie.match(/csrftoken=([^;]+)/)?.[1];
      const res = await fetch('/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken, 'Accept': 'text/event-stream' },
        body: JSON.stringify({ message: userMsg.content, conversation_id: activeConvId, agent: activeAgent, stream: true }),
      });

      if (res.headers.get('content-type')?.includes('text/event-stream')) {
        // Streaming response
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let full = '';
        let done = false;
        while (true) {
          const { done: streamDone, value } = await reader.read();
          if (streamDone) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                if (parsed.text) { full += parsed.text; setStreaming(full); }
                if (parsed.done) {
                  done = true;
                  setMessages((prev) => [...prev, { role: 'assistant', content: full }]);
                  setStreaming('');
                  if (parsed.conversation_id) { setActiveConvId(parsed.conversation_id); loadData(); }
                }
              } catch { full += data; setStreaming(full); }
            }
          }
        }
        if (full && !done) {
          setMessages((prev) => [...prev, { role: 'assistant', content: full }]);
        }
      } else {
        // Fallback: normal JSON response
        const data = await res.json();
        setMessages(data.messages || [...messages, userMsg, { role: 'assistant', content: data.response }]);
        setActiveAgent(data.active_agent);
        if (data.conversation_id && data.conversation_id !== activeConvId) {
          setActiveConvId(data.conversation_id);
          loadData();
        }
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `**Error:** ${err.message}` }]);
    } finally {
      setLoading(false);
      setStreaming('');
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleDelete = async (id, e) => {
    e?.stopPropagation();
    if (!confirm('Delete this conversation?')) return;
    try {
      await api.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConvId === id) { setActiveConvId(null); setMessages([]); }
    } catch (err) { console.error(err); }
  };

  const handlePin = async (id, e) => {
    e?.stopPropagation();
    try {
      const data = await api.togglePin(id);
      setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, is_pinned: data.is_pinned } : c)));
    } catch (err) { console.error(err); }
  };

  const handleRename = async (id) => {
    if (!renameTitle.trim()) return;
    try {
      const data = await api.renameConversation(id, renameTitle);
      setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, display_name: data.display_name } : c)));
      setRenameId(null);
    } catch (err) { console.error(err); }
  };

  const handleClear = async (id, e) => {
    e?.stopPropagation();
    if (!confirm('Clear all messages?')) return;
    try { await api.clearConversation(id); if (activeConvId === id) setMessages([]); }
    catch (err) { console.error(err); }
  };

  const handleVoiceResult = (text) => {
    setInput((prev) => prev ? prev + ' ' + text : text);
    inputRef.current?.focus();
  };

  const handleCommand = (cmd) => {
    switch (cmd.action) {
      case 'newChat': handleNewChat(); break;
      case 'clearChat': if (activeConvId) handleClear(activeConvId); break;
      case 'switchAgent':
      case 'switchChat':
        setActiveAgent(cmd.agent);
        api.switchAgent(cmd.agent).catch(() => {});
        break;
      case 'gotoTraining': navigate('/training'); break;
      case 'gotoSettings': navigate('/settings'); break;
      case 'exportMd': exportAsMarkdown(messages, activeConvId ? `Chat ${activeConvId}` : 'New Chat'); showToast('Exported as Markdown'); break;
      case 'exportJson': exportAsJson(messages, activeConvId ? `Chat ${activeConvId}` : 'New Chat'); showToast('Exported as JSON'); break;
    }
  };

  const handleFileUploaded = (file) => {
    showToast(`Uploaded: ${file.filename}`);
    if (file.content_text) {
      setInput((prev) => prev + `\n\n[File: ${file.filename}]\n${file.content_text.slice(0, 2000)}`);
    }
  };

  const handleLogout = async () => { try { await api.logout(); } catch {} window.location.href = '/login/'; };

  const sortedConversations = [...conversations].sort((a, b) => {
    if (a.is_pinned && !b.is_pinned) return -1;
    if (!a.is_pinned && b.is_pinned) return 1;
    return 0;
  });

  return (
    <div className="app-layout" onDragOver={(e) => { e.preventDefault(); if (e.dataTransfer.types.includes('Files')) setDragFiles(true); }}>
      {dragFiles && <DragDropUpload onFileUploaded={(f) => { if (f) handleFileUploaded(f); setDragFiles(false); }} />}

      {/* SIDEBAR */}
      <aside className="app-sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="logo-icon">AB</div>
            <div>
              <h1>Aura Book</h1>
              <span>AI Agent Platform</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <Link to="/" className="active">{t('chat')}</Link>
          <Link to="/training">{t('training')}</Link>
          <Link to="/settings">{t('settings')}</Link>
        </nav>

        <div className="sidebar-section">
          <h3>{t('conversations')}</h3>
          <button className="new-chat-btn" onClick={handleNewChat}>+ {t('newChat')}</button>
        </div>

        <div className="conversations-list">
          {sortedConversations.length === 0 ? (
            <div className="empty-state" style={{ padding: '30px 10px' }}>
              <div className="empty-icon">💬</div>
              <p>{t('noConversations')}</p>
            </div>
          ) : sortedConversations.map((conv) => (
            <div key={conv.id} className={`conversation-item ${activeConvId === conv.id ? 'active' : ''}`} onClick={() => selectConversation(conv)}>
              <div className="conv-icon" style={{ background: (AGENT_COLORS[conv.agent_name] || '#6c5ce7') + '22', color: AGENT_COLORS[conv.agent_name] || '#6c5ce7' }}>
                {AGENT_ICONS[conv.agent_name] || '💬'}
              </div>
              <div className="conv-info">
                {renameId === conv.id ? (
                  <input className="settings-input" value={renameTitle} onChange={(e) => setRenameTitle(e.target.value)}
                    onBlur={() => handleRename(conv.id)} onKeyDown={(e) => e.key === 'Enter' && handleRename(conv.id)}
                    autoFocus onClick={(e) => e.stopPropagation()} style={{ padding: '4px 8px', fontSize: '13px' }} />
                ) : <div className="conv-title">{conv.display_name}</div>}
                <div className="conv-agent">{conv.agent_name} {conv.is_pinned ? '📌' : ''}</div>
              </div>
              <div className="conv-actions">
                <button className="conv-action-btn" onClick={(e) => { e.stopPropagation(); setRenameId(conv.id); setRenameTitle(conv.display_name); }}>✏️</button>
                <button className="conv-action-btn" onClick={(e) => handlePin(conv.id, e)}>📌</button>
                <button className="conv-action-btn" onClick={(e) => handleClear(conv.id, e)}>🗑️</button>
                <button className="conv-action-btn danger" onClick={(e) => handleDelete(conv.id, e)}>✕</button>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-info">
            <div className="user-avatar">{userName?.charAt(0)?.toUpperCase() || 'U'}</div>
            <div className="user-details">
              <div className="user-name">{userName}</div>
              <div className="user-role">{t('activeAgent')}: {activeAgent}</div>
            </div>
          </div>
          <div className="sidebar-footer-actions">
            <ThemeSwitcher />
            <button className="footer-btn" onClick={() => setLang(lang === 'en' ? 'ar' : 'en')}>🌐 {lang === 'en' ? 'عربي' : 'EN'}</button>
            <button className="footer-btn" onClick={() => navigate('/settings')}>⚙️</button>
            <button className="footer-btn danger" onClick={handleLogout}>🚪</button>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="app-main">
        <div className="chat-header">
          <div className="agent-selector">
            {(agents.length ? agents : ['chat', 'code', 'data', 'research', 'planner', 'media']).map((a) => (
              <button key={a} className={`agent-chip ${activeAgent === a ? 'active' : ''}`}
                onClick={async () => { setActiveAgent(a); try { await api.switchAgent(a); } catch {} }}>
                {AGENT_ICONS[a]} {a}
              </button>
            ))}
          </div>
          <div className="chat-header-actions">
            <button className="header-action-btn" onClick={() => setCmdOpen(true)} title="Ctrl+K">🔍</button>
          </div>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && !streaming ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🤖</div>
              <h2>{t('welcome')}</h2>
              <p>{t('welcomeMessage')}</p>
              <div className="quick-actions">
                {['Explain quantum computing', 'Write a Python script', 'Analyze this dataset'].map((q) => (
                  <button key={q} className="quick-action-btn" onClick={() => { setInput(q); setTimeout(() => inputRef.current?.focus(), 0); }}>{q}</button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-avatar">{msg.role === 'user' ? '👤' : (AGENT_ICONS[activeAgent] || '🤖')}</div>
                <div className="message-body">
                  <div className="message-content">
                    <MessageContent content={msg.content} isUser={msg.role === 'user'} />
                  </div>
                  <MessageActions content={msg.content} isUser={msg.role === 'user'} />
                </div>
              </div>
            ))
          )}
          {streaming && (
            <div className="message assistant">
              <div className="message-avatar">{AGENT_ICONS[activeAgent] || '🤖'}</div>
              <div className="message-body">
                <div className="message-content">
                  <MessageContent content={streaming} isUser={false} />
                  <span className="streaming-cursor">|</span>
                </div>
              </div>
            </div>
          )}
          {loading && !streaming && (
            <div className="message assistant">
              <div className="message-avatar">{AGENT_ICONS[activeAgent] || '🤖'}</div>
              <div className="message-content">
                <div className="typing-indicator"><span></span><span></span><span></span></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <div className="chat-input-wrapper">
            <VoiceInput onResult={handleVoiceResult} disabled={loading} />
            <textarea ref={inputRef} className="chat-input" placeholder={t('typeMessage')} value={input}
              onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} rows={1} />
            <button className="send-btn" onClick={handleSend} disabled={loading || !input.trim()}>➤</button>
          </div>
          <div className="chat-input-hint">
            <span>💡 Ctrl+K for commands · Enter to send · Shift+Enter for new line</span>
          </div>
        </div>
      </main>

      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} onCommand={handleCommand} />
      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}

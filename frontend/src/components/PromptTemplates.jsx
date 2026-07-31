import { useState } from 'react';

const TEMPLATES = [
  {
    id: 'code-python',
    icon: '🐍',
    title: 'Write Python Code',
    category: 'Code',
    prompt: 'Write a Python function that {description}. Include docstrings and type hints.',
  },
  {
    id: 'code-javascript',
    icon: '⚡',
    title: 'Write JavaScript Code',
    category: 'Code',
    prompt: 'Write a JavaScript function that {description}. Use modern ES6+ syntax.',
  },
  {
    id: 'code-debug',
    icon: '🐛',
    title: 'Debug Code',
    category: 'Code',
    prompt: 'Help me debug this code:\n\n```\n{code}\n```\n\nThe error is: {error}',
  },
  {
    id: 'code-review',
    icon: '🔍',
    title: 'Code Review',
    category: 'Code',
    prompt: 'Review this code for best practices, performance issues, and potential bugs:\n\n```\n{code}\n```',
  },
  {
    id: 'data-analyze',
    icon: '📊',
    title: 'Analyze Data',
    category: 'Data',
    prompt: 'Analyze this dataset and provide insights:\n\n{data}\n\nInclude statistics, trends, and recommendations.',
  },
  {
    id: 'data-csv',
    icon: '📈',
    title: 'Process CSV',
    category: 'Data',
    prompt: 'Write Python code to process this CSV data:\n\n{csv_data}\n\nTasks: {tasks}',
  },
  {
    id: 'data-sql',
    icon: '🗃️',
    title: 'Write SQL Query',
    category: 'Data',
    prompt: 'Write a SQL query to {description}. Tables: {tables}',
  },
  {
    id: 'research-explain',
    icon: '📚',
    title: 'Explain Concept',
    category: 'Research',
    prompt: 'Explain {topic} in simple terms. Include examples and analogies.',
  },
  {
    id: 'research-compare',
    icon: '⚖️',
    title: 'Compare Options',
    category: 'Research',
    prompt: 'Compare {option1} vs {option2}. Include pros, cons, and recommendations.',
  },
  {
    id: 'research-summarize',
    icon: '📝',
    title: 'Summarize Text',
    category: 'Research',
    prompt: 'Summarize the following text in bullet points:\n\n{text}',
  },
  {
    id: 'plan-project',
    icon: '📋',
    title: 'Plan Project',
    category: 'Planning',
    prompt: 'Create a project plan for {project}. Include:\n1. Requirements\n2. Architecture\n3. Timeline\n4. Milestones',
  },
  {
    id: 'plan-todo',
    icon: '✅',
    title: 'Create TODO List',
    category: 'Planning',
    prompt: 'Break down "{task}" into actionable steps with priorities and estimated time.',
  },
  {
    id: 'write-email',
    icon: '✉️',
    title: 'Write Email',
    category: 'Writing',
    prompt: 'Write a professional email about {topic}. Tone: {tone}. To: {recipient}',
  },
  {
    id: 'write-blog',
    icon: '✍️',
    title: 'Write Blog Post',
    category: 'Writing',
    prompt: 'Write a blog post about {topic}. Include introduction, main points, and conclusion. Target audience: {audience}',
  },
  {
    id: 'translate',
    icon: '🌐',
    title: 'Translate Text',
    category: 'Writing',
    prompt: 'Translate the following text to {language}:\n\n{text}',
  },
  {
    id: 'image-gen',
    icon: '🎨',
    title: 'Generate Image Prompt',
    category: 'Creative',
    prompt: 'Create a detailed image generation prompt for: {description}. Style: {style}. Include composition, lighting, and mood.',
  },
];

const CATEGORIES = [...new Set(TEMPLATES.map(t => t.category))];

export default function PromptTemplates({ onSelect, onClose }) {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [search, setSearch] = useState('');
  const [customPrompt, setCustomPrompt] = useState('');

  const filtered = TEMPLATES.filter(t => {
    const matchCategory = selectedCategory === 'All' || t.category === selectedCategory;
    const matchSearch = t.title.toLowerCase().includes(search.toLowerCase()) ||
                        t.prompt.toLowerCase().includes(search.toLowerCase());
    return matchCategory && matchSearch;
  });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="prompt-templates-modal" onClick={e => e.stopPropagation()}>
        <div className="prompt-templates-header">
          <h3>📝 Prompt Templates</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="prompt-templates-search">
          <input
            type="text"
            placeholder="Search templates..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="prompt-search-input"
            autoFocus
          />
        </div>

        <div className="prompt-categories">
          <button
            className={`prompt-cat-btn ${selectedCategory === 'All' ? 'active' : ''}`}
            onClick={() => setSelectedCategory('All')}
          >All</button>
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              className={`prompt-cat-btn ${selectedCategory === cat ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat)}
            >{cat}</button>
          ))}
        </div>

        <div className="prompt-templates-list">
          {filtered.map(t => (
            <div key={t.id} className="prompt-template-item" onClick={() => { onSelect(t.prompt); onClose(); }}>
              <span className="prompt-template-icon">{t.icon}</span>
              <div className="prompt-template-info">
                <div className="prompt-template-title">{t.title}</div>
                <div className="prompt-template-preview">{t.prompt.slice(0, 60)}...</div>
              </div>
              <span className="prompt-template-category">{t.category}</span>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="prompt-empty">No templates found</div>
          )}
        </div>

        <div className="prompt-custom">
          <textarea
            placeholder="Or write your own prompt..."
            value={customPrompt}
            onChange={e => setCustomPrompt(e.target.value)}
            className="prompt-custom-input"
            rows={3}
          />
          <button
            className="prompt-custom-btn"
            disabled={!customPrompt.trim()}
            onClick={() => { onSelect(customPrompt); onClose(); }}
          >
            Use Custom Prompt
          </button>
        </div>
      </div>
    </div>
  );
}

export default function KeyboardShortcutsHelp({ onClose }) {
  const shortcuts = [
    { category: 'Navigation', items: [
      { keys: ['Ctrl', 'K'], desc: 'Command Palette' },
      { keys: ['Ctrl', 'P'], desc: 'Prompt Templates' },
      { keys: ['F11'], desc: 'Toggle Fullscreen' },
      { keys: ['Esc'], desc: 'Close Modal / Exit Fullscreen' },
    ]},
    { category: 'Chat', items: [
      { keys: ['Enter'], desc: 'Send Message' },
      { keys: ['Shift', 'Enter'], desc: 'New Line' },
      { keys: ['Ctrl', 'Shift', 'E'], desc: 'Export Chat' },
    ]},
    { category: 'Search', items: [
      { keys: ['Ctrl', 'F'], desc: 'Search Messages' },
      { keys: ['?'], desc: 'Show Shortcuts' },
    ]},
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="shortcuts-modal" onClick={e => e.stopPropagation()}>
        <div className="shortcuts-header">
          <h3>⌨️ Keyboard Shortcuts</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="shortcuts-body">
          {shortcuts.map(group => (
            <div key={group.category} className="shortcuts-group">
              <h4 className="shortcuts-group-title">{group.category}</h4>
              {group.items.map(item => (
                <div key={item.desc} className="shortcut-row">
                  <span className="shortcut-desc">{item.desc}</span>
                  <div className="shortcut-keys">
                    {item.keys.map((key, i) => (
                      <span key={i}>
                        <kbd className="shortcut-key">{key}</kbd>
                        {i < item.keys.length - 1 && <span className="shortcut-plus">+</span>}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="shortcuts-footer">
          <span>Press <kbd className="shortcut-key">?</kbd> anytime to show this</span>
        </div>
      </div>
    </div>
  );
}

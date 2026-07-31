import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useLang } from '../i18n/LanguageContext';
import { api } from '../api/client';

export default function RegisterPage() {
  const { t } = useLang();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password1, setPassword1] = useState('');
  const [password2, setPassword2] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (password1 !== password2) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const data = await api.register(username, password1, password2);
      if (data.status === 'ok') {
        window.location.href = '/';
      } else {
        setError(data.error || 'Registration failed');
      }
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <div className="auth-logo">AB</div>
            <h2>Create Account</h2>
            <p>Join Aura Book today</p>
          </div>

          {error && <div className="form-error">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Username</label>
              <input type="text" className="form-input" placeholder="Choose a username" value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input type="password" className="form-input" placeholder="Create a password" value={password1} onChange={(e) => setPassword1(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>Confirm Password</label>
              <input type="password" className="form-input" placeholder="Confirm your password" value={password2} onChange={(e) => setPassword2(e.target.value)} required />
            </div>
            <button type="submit" className="form-btn" disabled={loading}>
              {loading ? t('loading') : t('register')}
            </button>
          </form>

          <div className="auth-footer">
            Already have an account? <Link to="/login">{t('login')}</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useLang } from '../i18n/LanguageContext';
import { api } from '../api/client';

export default function TrainingPage() {
  const { t } = useLang();
  const [jobs, setJobs] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({
    model_name: 'google/gemma-2-2b-it',
    dataset_size: '100k',
    batch_size: 10000,
    email: '',
    notify_on_complete: true,
  });
  const [toast, setToast] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    loadJobs();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const loadJobs = async () => {
    try {
      const data = await api.trainingDashboard();
      setJobs(data.jobs || []);
      // Start polling for running jobs
      (data.jobs || []).forEach((j) => {
        if (j.status === 'running') startPolling(j.id);
      });
    } catch {
      // Use empty state
    }
  };

  const startPolling = (jobId) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const data = await api.trainingStatus(jobId);
        setJobs((prev) => prev.map((j) => (j.id === jobId ? { ...j, ...data } : j)));
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 5000);
  };

  const handleStart = async () => {
    try {
      const data = await api.startTraining(form);
      const newJob = {
        id: data.id,
        model_name: form.model_name,
        dataset_size: form.dataset_size,
        email: form.email,
        status: 'pending',
        current_round: 0,
        total_rounds: 10,
        progress: 0,
      };
      setJobs((prev) => [newJob, ...prev]);
      setShowModal(false);
      showToast(t('running'), 'success');
      startPolling(data.id);
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  const handleStop = async (jobId) => {
    try {
      await api.stopTraining(jobId);
      setJobs((prev) => prev.map((j) => (j.id === jobId ? { ...j, status: 'failed', error_message: 'Stopped by user' } : j)));
      showToast('Training stopped', 'success');
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const stats = {
    total: jobs.length,
    running: jobs.filter((j) => j.status === 'running').length,
    completed: jobs.filter((j) => j.status === 'completed').length,
    failed: jobs.filter((j) => j.status === 'failed').length,
  };

  return (
    <div className="training-page">
      <div className="training-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Link to="/" style={{ color: 'var(--text-secondary)', textDecoration: 'none', fontSize: '14px' }}>
            ← {t('chat')}
          </Link>
          <h1>{t('trainingDashboard')}</h1>
        </div>
        <button className="start-training-btn" onClick={() => setShowModal(true)}>
          + {t('startTraining')}
        </button>
      </div>

      <div className="training-content">
        <div className="training-stats">
          <div className="stat-card">
            <div className="stat-icon purple">📊</div>
            <div className="stat-info">
              <h3>{stats.total}</h3>
              <p>{t('trainingHistory')}</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon yellow">⚡</div>
            <div className="stat-info">
              <h3>{stats.running}</h3>
              <p>{t('running')}</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green">✅</div>
            <div className="stat-info">
              <h3>{stats.completed}</h3>
              <p>{t('completed')}</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon red">❌</div>
            <div className="stat-info">
              <h3>{stats.failed}</h3>
              <p>{t('failed')}</p>
            </div>
          </div>
        </div>

        <div className="training-jobs">
          <div className="training-jobs-header">{t('trainingHistory')}</div>
          {jobs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🚀</div>
              <h3>{t('noTrainingJobs')}</h3>
              <p>Start your first training job</p>
            </div>
          ) : (
            jobs.map((job) => (
              <div key={job.id} className="training-job-item">
                <div className="job-info">
                  <h4>{job.model_name}</h4>
                  <p>
                    {t('datasetSize')}: {job.dataset_size || job.total_samples} · {t('email')}: {job.email || 'N/A'}
                  </p>
                </div>
                <span className={`job-status ${job.status}`}>{t(job.status)}</span>
                <div className="job-progress">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${job.progress || 0}%` }}
                    />
                  </div>
                  <div className="progress-text">
                    {job.current_loss ? `${t('loss')}: ${job.current_loss.toFixed(4)}` : `${job.progress || 0}%`}
                  </div>
                </div>
                <div className="job-actions">
                  {(job.status === 'running' || job.status === 'pending') && (
                    <button className="job-action-btn danger" onClick={() => handleStop(job.id)}>
                      {t('stopTraining')}
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{t('startTraining')}</h3>

            <div className="form-group">
              <label>{t('model')}</label>
              <select className="settings-select" value={form.model_name} onChange={(e) => setForm({ ...form, model_name: e.target.value })}>
                <option value="google/gemma-2-2b-it">Google Gemma-2-2B-it</option>
                <option value="google/gemma-2-9b-it">Google Gemma-2-9B-it</option>
                <option value="meta-llama/Llama-3-8B">Meta Llama-3-8B</option>
                <option value="Qwen/Qwen2-7B">Qwen2-7B</option>
              </select>
            </div>

            <div className="form-group">
              <label>{t('datasetSize')}</label>
              <select className="settings-select" value={form.dataset_size} onChange={(e) => setForm({ ...form, dataset_size: e.target.value })}>
                <option value="10k">10k</option>
                <option value="50k">50k</option>
                <option value="100k">100k</option>
                <option value="200k">200k</option>
                <option value="500k">500k</option>
              </select>
            </div>

            <div className="form-group">
              <label>{t('batchSize')}</label>
              <input type="number" className="form-input" value={form.batch_size} onChange={(e) => setForm({ ...form, batch_size: parseInt(e.target.value) || 10000 })} />
            </div>

            <div className="form-group">
              <label>{t('email')}</label>
              <input type="email" className="form-input" placeholder="your@email.com" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
            </div>

            <div className="settings-toggle">
              <span style={{ fontSize: '14px' }}>{t('notifyOnComplete')}</span>
              <div
                className={`toggle-switch ${form.notify_on_complete ? 'active' : ''}`}
                onClick={() => setForm({ ...form, notify_on_complete: !form.notify_on_complete })}
              />
            </div>

            <div className="modal-actions">
              <button className="modal-btn" onClick={() => setShowModal(false)}>{t('cancel')}</button>
              <button className="modal-btn primary" onClick={handleStart}>{t('startTraining')}</button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}

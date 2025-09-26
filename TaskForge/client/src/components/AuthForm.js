import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { login, register } from '../services/authService';

const AuthForm = () => {
  const navigate = useNavigate();
  const { login: setAuthToken } = useAuth();
  const [mode, setMode] = useState('login');
  const [formState, setFormState] = useState({ email: '', password: '' });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormState((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const { token } = mode === 'login' ? await login(formState) : await register(formState);
      setAuthToken(token);
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.response?.data?.message || 'Помилка авторизації');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-form">
      <h2>{mode === 'login' ? 'Вхід' : 'Реєстрація'}</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          name="email"
          type="email"
          value={formState.email}
          onChange={handleChange}
          required
        />

        <label htmlFor="password">Пароль</label>
        <input
          id="password"
          name="password"
          type="password"
          minLength={6}
          value={formState.password}
          onChange={handleChange}
          required
        />

        <button type="submit" disabled={loading}>
          {loading ? 'Обробка...' : mode === 'login' ? 'Увійти' : 'Зареєструватися'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      <button type="button" className="switch" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
        {mode === 'login' ? 'Немає акаунту? Зареєструйтесь' : 'Вже маєте акаунт? Увійдіть'}
      </button>
    </div>
  );
};

export default AuthForm;

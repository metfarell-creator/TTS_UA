import React from 'react';
import { Navigate } from 'react-router-dom';
import AuthForm from '../components/AuthForm';
import { useAuth } from '../context/AuthContext';

const AuthPage = () => {
  const { token } = useAuth();

  if (token) {
    return <Navigate to="/" replace />;
  }

  return (
    <main className="auth-page">
      <section>
        <h1>TaskForge</h1>
        <p>Керуйте своїми завданнями швидко та безпечно.</p>
      </section>
      <AuthForm />
    </main>
  );
};

export default AuthPage;

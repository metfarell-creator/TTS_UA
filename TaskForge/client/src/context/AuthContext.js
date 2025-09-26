import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { getProfileFromToken } from '../services/authService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem('taskforge_token'));
  const [user, setUser] = useState(() => (token ? getProfileFromToken(token) : null));

  useEffect(() => {
    if (token) {
      localStorage.setItem('taskforge_token', token);
    } else {
      localStorage.removeItem('taskforge_token');
    }
  }, [token]);

  const value = useMemo(
    () => ({
      token,
      user,
      login: (authToken) => {
        setToken(authToken);
        setUser(getProfileFromToken(authToken));
      },
      logout: () => {
        setToken(null);
        setUser(null);
      }
    }),
    [token, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

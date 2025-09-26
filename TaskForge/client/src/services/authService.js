import apiClient from './apiClient';

export const register = async ({ email, password }) => {
  const { data } = await apiClient.post('/auth/register', { email, password });
  return data;
};

export const login = async ({ email, password }) => {
  const { data } = await apiClient.post('/auth/login', { email, password });
  return data;
};

export const getProfileFromToken = (token) => {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return { id: payload.id, email: payload.email };
  } catch (error) {
    return null;
  }
};

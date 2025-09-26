import axios from 'axios';

const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const apiClient = axios.create({
  baseURL: apiBaseUrl
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('taskforge_token');
  if (token) {
    // eslint-disable-next-line no-param-reassign
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default apiClient;

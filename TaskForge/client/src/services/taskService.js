import apiClient from './apiClient';

export const getTasks = async () => {
  const { data } = await apiClient.get('/tasks');
  return data;
};

export const createTask = async (payload) => {
  const { data } = await apiClient.post('/tasks', payload);
  return data;
};

export const updateTask = async (id, payload) => {
  const { data } = await apiClient.put(`/tasks/${id}`, payload);
  return data;
};

export const deleteTask = async (id) => {
  await apiClient.delete(`/tasks/${id}`);
};

import React, { useEffect, useMemo, useState } from 'react';
import TaskForm from '../components/TaskForm';
import TaskList from '../components/TaskList';
import { useAuth } from '../context/AuthContext';
import { createTask, deleteTask, getTasks, updateTask } from '../services/taskService';

const DashboardPage = () => {
  const { user, logout } = useAuth();
  const [tasks, setTasks] = useState([]);
  const [activeTask, setActiveTask] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const completedCount = useMemo(() => tasks.filter((task) => task.completed).length, [tasks]);

  const refreshTasks = async () => {
    try {
      setLoading(true);
      const data = await getTasks();
      setTasks(data);
    } catch (err) {
      setError('Не вдалося завантажити завдання');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshTasks();
  }, []);

  const handleCreateTask = async (task) => {
    setError(null);
    try {
      const newTask = await createTask(task);
      setTasks((prev) => [newTask, ...prev]);
      setActiveTask(null);
    } catch (err) {
      setError('Не вдалося створити завдання');
    }
  };

  const handleUpdateTask = async (task) => {
    setError(null);
    try {
      const updated = await updateTask(activeTask.id, task);
      setTasks((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
      setActiveTask(null);
    } catch (err) {
      setError('Не вдалося оновити завдання');
    }
  };

  const handleDeleteTask = async (task) => {
    setError(null);
    try {
      await deleteTask(task.id);
      setTasks((prev) => prev.filter((item) => item.id !== task.id));
    } catch (err) {
      setError('Не вдалося видалити завдання');
    }
  };

  const handleToggleComplete = async (task) => {
    setError(null);
    try {
      const updated = await updateTask(task.id, { completed: !task.completed });
      setTasks((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
    } catch (err) {
      setError('Не вдалося оновити статус завдання');
    }
  };

  return (
    <main className="dashboard">
      <header>
        <div>
          <h1>Привіт, {user?.email}</h1>
          <p>
            Виконано {completedCount} з {tasks.length}
          </p>
        </div>
        <button type="button" onClick={logout}>
          Вийти
        </button>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="content">
        <div className="left">
          <TaskForm
            initialTask={activeTask}
            onSubmit={activeTask ? handleUpdateTask : handleCreateTask}
            onCancel={() => setActiveTask(null)}
          />
        </div>
        <div className="right">
          {loading ? <p>Завантаження...</p> : <TaskList tasks={tasks} onToggleComplete={handleToggleComplete} onEdit={setActiveTask} onDelete={handleDeleteTask} />}
        </div>
      </section>
    </main>
  );
};

export default DashboardPage;

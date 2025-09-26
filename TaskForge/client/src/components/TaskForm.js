import React, { useEffect, useState } from 'react';

const defaultState = { title: '', description: '' };

const TaskForm = ({ initialTask, onSubmit, onCancel }) => {
  const [task, setTask] = useState(defaultState);

  useEffect(() => {
    if (initialTask) {
      setTask({ title: initialTask.title, description: initialTask.description || '' });
    } else {
      setTask(defaultState);
    }
  }, [initialTask]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setTask((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(task);
  };

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      <h3>{initialTask ? 'Редагувати завдання' : 'Нове завдання'}</h3>
      <label htmlFor="title">Назва</label>
      <input id="title" name="title" value={task.title} onChange={handleChange} required />

      <label htmlFor="description">Опис</label>
      <textarea id="description" name="description" value={task.description} onChange={handleChange} />

      <div className="actions">
        <button type="submit">{initialTask ? 'Зберегти' : 'Створити'}</button>
        {onCancel && (
          <button type="button" onClick={onCancel} className="secondary">
            Скасувати
          </button>
        )}
      </div>
    </form>
  );
};

export default TaskForm;

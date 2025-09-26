import React from 'react';

const TaskList = ({ tasks, onToggleComplete, onEdit, onDelete }) => {
  if (!tasks.length) {
    return <p>У вас ще немає завдань. Додайте перше!</p>;
  }

  return (
    <ul className="task-list">
      {tasks.map((task) => (
        <li key={task.id} className={task.completed ? 'completed' : ''}>
          <div>
            <h4>{task.title}</h4>
            {task.description && <p>{task.description}</p>}
          </div>
          <div className="task-actions">
            <button type="button" onClick={() => onToggleComplete(task)}>
              {task.completed ? 'Відновити' : 'Завершити'}
            </button>
            <button type="button" onClick={() => onEdit(task)}>
              Редагувати
            </button>
            <button type="button" className="danger" onClick={() => onDelete(task)}>
              Видалити
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
};

export default TaskList;

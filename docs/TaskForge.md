# TaskForge: Архітектура, код та розгортання

## I. Резюме та архітектурний план проекту

### 1.1. Вступ до проекту
"TaskForge" — це повнофункціональний односторінковий веб-додаток (SPA) для управління персональними завданнями. Він надає користувачам інтуїтивно зрозумілий інтерфейс для реєстрації, автентифікації та керування списками завдань у безпечному середовищі. Проект реалізований з використанням сучасних інженерних практик і служить як готове рішення, так і освітній ресурс для вивчення розробки програмного забезпечення.

**Основні функціональні можливості:**

- **Автентифікація користувачів:** система реєстрації та входу з хешуванням паролів (bcrypt) і керуванням сесій через JWT.
- **Управління завданнями (CRUD):**
  - створення завдань із заголовком та описом;
  - перегляд списку завдань;
  - редагування тексту та статусу завдань;
  - видалення завдань.
- **Авторизація та ізоляція даних:** гарантується доступ користувача виключно до власних даних; API блокує спроби доступу до чужих завдань.

### 1.2. Вибір технологічного стеку: PERN
Для реалізації "TaskForge" обрано стек PERN (PostgreSQL, Express.js, React, Node.js), який поєднує потужність реляційної бази даних із гнучкістю JavaScript-серверного та клієнтського коду.

- **PostgreSQL:** об'єктно-реляційна СУБД з відкритим вихідним кодом, що забезпечує високу надійність, відповідність стандартам SQL та гарантії цілісності даних (ACID).
- **Express.js:** мінімалістичний веб-фреймворк для Node.js, який надає інструменти для створення RESTful API, управління маршрутизацією та обробкою HTTP-запитів.
- **React:** бібліотека для створення користувацьких інтерфейсів із компонентною архітектурою.
- **Node.js:** середовище виконання JavaScript на сервері, що забезпечує швидкі та масштабовані мережеві додатки.

Реляційна модель PostgreSQL краще підходить для додатків, де цілісність даних є пріоритетною. Зовнішні ключі гарантують, що завдання не можуть існувати без прив'язки до користувача, запобігаючи появі «осиротілих» записів.

### 1.3. Високорівнева архітектура системи
"TaskForge" побудований за трирівневою архітектурою, яка чітко розділяє відповідальності між клієнтом, сервером та базою даних.

- **Клієнт (Frontend):** односторінковий додаток на React, що відповідає за відображення даних та взаємодію з користувачем.
- **Сервер (Backend):** RESTful API на Node.js та Express.js, який реалізує бізнес-логіку, валідує дані та взаємодіє з базою даних.
- **База даних (Persistence Layer):** PostgreSQL, яка зберігає дані користувачів та завдань у чітко визначених таблицях із зв'язками між ними.

## II. Детальний опис кодової бази

### 2.1. Структура проекту
Проект поділений на дві основні директорії: `client` (фронтенд) та `server` (бекенд).

```
TaskForge/
├── client/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── server/
│   ├── config/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   ├── routes/
│   ├── app.js
│   └── package.json
└── README.md
```

### 2.2. Фронтенд (React)
Фронтенд реалізований як SPA на React з використанням функціональних компонентів та хуків. Основні компоненти включають:

- `AuthForm`: форма реєстрації та входу.
- `TaskList`: список завдань користувача.
- `TaskForm`: форма для створення та редагування завдань.

```jsx
import React, { useEffect, useState } from 'react';
import { getTasks } from '../services/taskService';

const TaskList = () => {
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    const fetchTasks = async () => {
      const data = await getTasks();
      setTasks(data);
    };
    fetchTasks();
  }, []);

  return (
    <div>
      <h2>Список завдань</h2>
      <ul>
        {tasks.map(task => (
          <li key={task.id}>
            {task.title} - {task.completed ? 'Виконано' : 'Не виконано'}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TaskList;
```

### 2.3. Бекенд (Node.js + Express.js)
Бекенд реалізований як RESTful API з використанням Express.js та PostgreSQL. Основні модулі включають:

- `models/task.js`: модель завдань для взаємодії з базою даних.
- `controllers/taskController.js`: контролери для обробки запитів CRUD.
- `routes/taskRoutes.js`: маршрути для завдань.
- `middleware/auth.js`: мідлвар для перевірки JWT.

```javascript
const { Sequelize, DataTypes } = require('sequelize');
const sequelize = require('../config/db');

const Task = sequelize.define('Task', {
  title: {
    type: DataTypes.STRING,
    allowNull: false
  },
  description: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  completed: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  userId: {
    type: DataTypes.INTEGER,
    allowNull: false
  }
});

module.exports = Task;
```

```javascript
const Task = require('../models/task');

exports.createTask = async (req, res) => {
  try {
    const { title, description } = req.body;
    const task = await Task.create({
      title,
      description,
      userId: req.user.id
    });
    res.status(201).json(task);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

exports.getTasks = async (req, res) => {
  try {
    const tasks = await Task.findAll({ where: { userId: req.user.id } });
    res.json(tasks);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```

## III. Інструкції з розгортання

### 3.1. Вимоги до середовища

- Node.js (версія 14+)
- PostgreSQL (версія 12+)
- npm або yarn

### 3.2. Кроки розгортання

1. **Встановлення залежностей**

   ```bash
   cd client
   npm install

   cd ../server
   npm install
   ```

2. **Налаштування бази даних**
   - Створіть базу даних PostgreSQL та налаштуйте з'єднання у файлі `server/config/db.js`.
   - Запустіть міграції (якщо використовуєте Sequelize) або створіть таблиці вручну.

3. **Налаштування змінних середовища**

   Створіть файл `.env` у директорії `server` з наступними змінними:

   ```env
   JWT_SECRET=your_jwt_secret
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

4. **Запуск додатку**

   ```bash
   cd server
   npm start
   ```

   В іншому терміналі запустіть клієнт:

   ```bash
   cd client
   npm start
   ```

5. **Доступ до додатку**

   Відкрийте браузер та перейдіть за адресою `http://localhost:3000`.

## IV. Висновок
"TaskForge" — це повнофункціональний додаток для управління завданнями, реалізований з використанням сучасних технологій та інженерних практик. Проект можна розгорнути як для персонального користування, так і як освітній приклад для вивчення веб-розробки.

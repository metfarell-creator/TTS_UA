const { validationResult } = require('express-validator');
const Task = require('../models/Task');

exports.createTask = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  const { title, description } = req.body;

  try {
    const task = await Task.create({
      title,
      description,
      userId: req.user.id
    });

    res.status(201).json(task);
  } catch (error) {
    res.status(500).json({ message: 'Could not create task', error: error.message });
  }
};

exports.getTasks = async (req, res) => {
  try {
    const tasks = await Task.findAll({ where: { userId: req.user.id }, order: [['createdAt', 'DESC']] });
    res.json(tasks);
  } catch (error) {
    res.status(500).json({ message: 'Could not fetch tasks', error: error.message });
  }
};

exports.getTaskById = async (req, res) => {
  try {
    const task = await Task.findOne({ where: { id: req.params.id, userId: req.user.id } });
    if (!task) {
      return res.status(404).json({ message: 'Task not found' });
    }

    res.json(task);
  } catch (error) {
    res.status(500).json({ message: 'Could not fetch task', error: error.message });
  }
};

exports.updateTask = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    const task = await Task.findOne({ where: { id: req.params.id, userId: req.user.id } });
    if (!task) {
      return res.status(404).json({ message: 'Task not found' });
    }

    const { title, description, completed } = req.body;
    task.title = title ?? task.title;
    task.description = description ?? task.description;
    if (typeof completed === 'boolean') {
      task.completed = completed;
    }

    await task.save();

    res.json(task);
  } catch (error) {
    res.status(500).json({ message: 'Could not update task', error: error.message });
  }
};

exports.deleteTask = async (req, res) => {
  try {
    const task = await Task.findOne({ where: { id: req.params.id, userId: req.user.id } });
    if (!task) {
      return res.status(404).json({ message: 'Task not found' });
    }

    await task.destroy();
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ message: 'Could not delete task', error: error.message });
  }
};

const { Router } = require('express');
const { body, param } = require('express-validator');
const taskController = require('../controllers/taskController');
const auth = require('../middleware/auth');

const router = Router();

router.use(auth);

const idParamValidator = param('id').isInt().toInt();

router.post(
  '/',
  [body('title').notEmpty().withMessage('Title is required'), body('description').optional().isString()],
  taskController.createTask
);

router.get('/', taskController.getTasks);
router.get('/:id', [idParamValidator], taskController.getTaskById);
router.put(
  '/:id',
  [
    idParamValidator,
    body('title').optional().isString(),
    body('description').optional().isString(),
    body('completed').optional().isBoolean()
  ],
  taskController.updateTask
);
router.delete('/:id', [idParamValidator], taskController.deleteTask);

module.exports = router;

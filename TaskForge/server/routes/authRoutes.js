const { Router } = require('express');
const { body } = require('express-validator');
const authController = require('../controllers/authController');

const router = Router();

const emailValidator = body('email').isEmail().withMessage('Valid email is required');
const passwordValidator = body('password')
  .isLength({ min: 6 })
  .withMessage('Password must be at least 6 characters long');

router.post('/register', [emailValidator, passwordValidator], authController.register);
router.post('/login', [emailValidator, passwordValidator], authController.login);

module.exports = router;

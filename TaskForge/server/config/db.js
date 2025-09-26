const { Sequelize } = require('sequelize');
const dotenv = require('dotenv');

dotenv.config();

const {
  DB_NAME,
  DB_USER,
  DB_PASSWORD,
  DB_HOST = 'localhost',
  DB_PORT = '5432',
  DB_LOGGING = 'false'
} = process.env;

const sequelize = new Sequelize(DB_NAME, DB_USER, DB_PASSWORD, {
  host: DB_HOST,
  port: DB_PORT,
  dialect: 'postgres',
  logging: DB_LOGGING === 'true' ? console.log : false,
  define: {
    underscored: true
  }
});

module.exports = sequelize;

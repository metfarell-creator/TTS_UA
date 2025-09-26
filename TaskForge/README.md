# TaskForge

TaskForge is a full-stack task management application built on the PERN stack (PostgreSQL, Express.js, React, Node.js). The project demonstrates a secure authentication flow, a protected REST API, and a responsive single-page interface for managing personal tasks.

## Project Structure

```
TaskForge/
├── client/               # React single-page application
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── context/
│       ├── App.js
│       └── index.js
├── server/               # Express + Sequelize REST API
│   ├── config/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   └── routes/
└── README.md
```

## Getting Started

### Prerequisites
- Node.js 14+
- PostgreSQL 12+

### Environment Variables
Copy the server `.env.example` to `.env` and edit as needed:

```
cd server
cp .env.example .env
```

### Quick Start with Docker Compose

The fastest way to spin up the entire stack (PostgreSQL, API, and React UI) is with Docker Compose:

```bash
docker compose up --build
```

The services are exposed as follows:

- React UI: [http://localhost:3000](http://localhost:3000)
- REST API: [http://localhost:5000/api](http://localhost:5000/api)
- PostgreSQL: `localhost:5432` (database `taskforge`, user/password `taskforge`)

To stop the stack press `Ctrl+C` and run `docker compose down` to remove containers. Database data is stored in a named volume `taskforge_db_data`.

### Install Dependencies

```
# Install server dependencies
cd server
npm install

# Install client dependencies
cd ../client
npm install
```

### Run the Application

Start the backend API:

```
cd server
npm run dev
```

In a separate terminal start the frontend:

```
cd client
npm start
```

The React app runs on [http://localhost:3000](http://localhost:3000) and proxies API calls to the backend at `http://localhost:5000` by default.

## Docker Images

- `client/Dockerfile` builds a production-ready bundle served by Nginx.
- `server/Dockerfile` installs API dependencies and starts the Express server.
- `docker-compose.yml` orchestrates the database, API, and UI services for local development or lightweight deployment.

## Database Initialization

The server automatically synchronizes the Sequelize models with the database schema on startup. For production systems, consider using migrations instead of `sequelize.sync()`.

## Scripts

### Server
- `npm start` – start the API
- `npm run dev` – start the API with auto-reload (requires nodemon)
- `npm run lint` – lint the codebase with ESLint

### Client
- `npm start` – run the React development server
- `npm run build` – create an optimized production build
- `npm test` – run the CRA test runner

## License

TaskForge is distributed under the MIT License.

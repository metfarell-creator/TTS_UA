FROM node:18-alpine AS build

WORKDIR /app

COPY package*.json ./

RUN npm ci

ARG REACT_APP_API_URL=http://localhost:5000/api
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

COPY . .

RUN npm run build

FROM nginx:1.25-alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

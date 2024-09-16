# build stage
FROM node:lts-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# production stage
FROM oven/bun:1 as production-stage
WORKDIR /usr/src/app
COPY --from=build-stage /app/.output .
EXPOSE 80
CMD ["bun", ".output/server/index.mjs"]
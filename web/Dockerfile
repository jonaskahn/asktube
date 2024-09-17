# build stage
FROM node:lts-bookworm AS build-stage
WORKDIR /src
RUN npm install -g pnpm
COPY package.json .
RUN pnpm install
COPY . .
RUN pnpm run build

# production stage
FROM oven/bun:1 AS production-stage
WORKDIR /usr/src/app
COPY --from=build-stage /src/.output .
EXPOSE 3000

ENV NUXT_PUBLIC_API_URL=''
CMD ["bun", "./server/index.mjs"]
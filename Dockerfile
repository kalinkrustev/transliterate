FROM node:10.15.3-alpine AS build
RUN set -xe \
    && apk add --no-cache bash git openssh python make g++ \
    && git --version && bash --version && ssh -V && npm -v && node -v && yarn -v
WORKDIR /app
COPY package.json package.json
RUN npm install
COPY . .

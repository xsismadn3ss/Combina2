# syntax=docker/dockerfile:1
# Combina2 frontend - multi-stage standalone production image
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# Cache npm via BuildKit
RUN --mount=type=cache,target=/root/.npm npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# El navegador utiliza el mismo origen y Next proxya las peticiones al backend.
ARG NEXT_PUBLIC_API_URL=http://localhost:3000
ARG BACKEND_INTERNAL_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV BACKEND_INTERNAL_URL=$BACKEND_INTERNAL_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
# Copia artefactos de build + deps de produccion
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]

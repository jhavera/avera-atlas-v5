# =============================================================================
# AVERA-ATLAS Demo Service
# Multi-stage Docker build for production deployment
# =============================================================================

# Stage 1: Build the React application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the production bundle
RUN npm run build

# =============================================================================
# Stage 2: Production image with nginx
# =============================================================================
FROM nginx:alpine AS production

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

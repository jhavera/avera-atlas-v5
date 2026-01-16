# AVERA-ATLAS Demo Service

Interactive 3D visualization and presentation for the AVERA-ATLAS Space Debris Detection & Conjunction Assessment System.

## Features

- **3D Orbital Visualization**: Multi-sensor acquisition demonstration with three CubeSats tracking debris
- **Animated Pipeline Diagram**: APS data flow from Ingest → Detect → Track → Propagate → Assess → Decide
- **Architecture Overview**: Microservices architecture display with service status
- **Risk Classification**: Real-time RED/AMBER/GREEN risk level display

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run production container
docker-compose up -d demo

# Access at http://localhost:8090
```

### Using Docker directly

```bash
# Build the image
docker build -t avera-atlas-demo:latest .

# Run the container
docker run -d -p 8090:80 --name avera-atlas-demo avera-atlas-demo:latest

# Access at http://localhost:8090
```

### Development Mode

```bash
# Option 1: Docker Compose with hot reload
docker-compose --profile dev up demo-dev

# Option 2: Local development
npm install
npm run dev

# Access at http://localhost:3000
```

## Integration with AVERA-ATLAS

To integrate with your existing AVERA-ATLAS services, add this service to your main `docker-compose.yml`:

```yaml
services:
  # ... your existing services (ingest, detector, propagator, viz, ui)
  
  demo:
    build:
      context: ./demo
      dockerfile: Dockerfile
    ports:
      - "8090:80"
    networks:
      - avera-network
```

## Project Structure

```
demo/
├── src/
│   ├── App.jsx          # Main visualization component
│   └── main.jsx         # React entry point
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
├── Dockerfile           # Multi-stage production build
├── docker-compose.yml   # Container orchestration
├── nginx.conf           # Production server config
└── README.md            # This file
```

## Views

| View | Description |
|------|-------------|
| **Overview** | System architecture, microservices grid, current risk assessment |
| **Orbital** | 3D visualization of multi-sensor debris tracking (ATLAS-1/2/3 CubeSats) |
| **Pipeline** | Animated APS data flow diagram with stage progression |

## Service Port Reference

| Service | Port | Description |
|---------|------|-------------|
| ingest | 8001 | CubeSat SWIR data acquisition |
| detector | 8002 | YOLOv8 object detection |
| tracker | 8003 | Multi-sensor IOD (planned) |
| propagator | 8004 | Keplerian orbit propagation |
| viz | 8005 | Operational 3D visualization |
| ui | 8080 | Web dashboard interface |
| **demo** | **8090** | **Interactive presentation** |

## Health Check

The container includes a health check endpoint:

```bash
curl http://localhost:8090/health
# Returns: healthy
```

## License

Part of the AVERA-ATLAS project.

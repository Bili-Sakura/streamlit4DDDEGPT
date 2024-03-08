# DDE-GPT

## Quick Start

**1. Initalizaiton**  
```
docker-compose up
```

**2. Update(Option)**  

modify `docker-compose.yml`
```yaml
# docker-compose.yml
version: '3.8'
services:
  ddegpt: 
    build: Dockerfile.dev # for fev
    ports:
      - "8503:8503" # self-defined
```

```
docker-compose up
```
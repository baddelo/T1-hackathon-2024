# T1-hackathon-2024

## Описание проекта

#### Технологии

#### Метрики


## Как запустить?

### 

### Только модель (notebook)

### Проект целиком (frontend + backend)

Системные требования: виртуальная машина на Ubuntu 22.04 LTS с Nvidia GPU

### Установить docker

```shell
curl -fsSL https://get.docker.com -o install-docker.sh
```
```shell
sudo sh install-docker.sh
```


### Установить NVIDIA Container Toolkit 

```shell

```
### Уcтановить cuda

#### Склонировать репозиторий

```shell
git clone https://github.com/baddelo/T1-hackathon-2024.git
```

####

#### Запуск контейнеров проекта
```shell
docker compose -f docker-compose.yml up --build
```

#### Остановка контейнеров проекта
```shell
docker compose -f docker-compose.yml down
```

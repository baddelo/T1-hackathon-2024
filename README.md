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
```shell
dockerd-rootless-setuptool.sh install
```

### Установить NVIDIA Container Toolkit 

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```shell
sudo apt-get update
```

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

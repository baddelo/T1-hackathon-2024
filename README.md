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
sudo apt-get install -y nvidia-container-toolkit
```

### Уcтановить cuda

```bash
sudo apt-key del 7fa2af80
```
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```
```shell
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```
```shell
sudo apt-get update
```
```shell
sudo apt-get -y install cuda-toolkit-12-6
```

#### Склонировать репозиторий

```shell
git clone https://github.com/baddelo/T1-hackathon-2024.git
```
```shell
cd T1-hackathon-2024/docker
```

#### Загрузить актуальный докер-образ

(самостоятельная сборка не предлагается из-за несовместимости версий poetry для python 3.12, который использовался при разработке, с poetry для python 3.10 на ubuntu 22.04)

```shell
curl -fsSL 'https://drive.usercontent.google.com/download?id=1FSNn5oQvlUOXqU1fCc4MPdDVUQKXYHrF&export=download&authuser=0&confirm=t&uuid=35cebddf-8996-4864-91ff-d3eb87c20774&at=AN_67v2rXFM5nvCt0oVSGEH7w7eQ%3A1730030521756' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8' -H 'Accept-Language: ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'DNT: 1' -H 'Alt-Used: drive.usercontent.google.com' -H 'Connection: keep-alive' -H 'Referer: https://drive.usercontent.google.com/' -H 'Cookie: AEC=AVYB7cpH6RndIR5XyVp5u5IsQGljBuMfalVOu1bcLjms5AH9Bt7RP8L8sw; NID=518=IQXw-g7gFRYFaVuH3CrfrkIi3Z712_x_A8VTou2WUA9_PFZd7z1bi5-PGjmWSu0w4dNoCFHYXulTzS3VQnI0win48UWuB0Z2-K_Gzny5nvMK5t8CpKa2i6xFmpltvJExHkVOtPrHlIKCXJob4EWUDGeXm51CMqx_P6TMBTfhGCl8nb4oQiCwXVLKL7ai96jldbzRfShrBN5F1eiAILyMrUHR-lARkOPSNBoyeZGLJQWYw670nuzA-PApXSM7vIVkzeA44GEc5Q41yh6bLj_1A8wIuJAsxW2jEl8nWcQaBRIYb8UnIYLh7ygkqPEgAgeTQvE_onYe7wg0_pG7IOAame9dK_LzYI325ndxeKEzVRwijzJmmrbSaAUx3kvZP5437DUpGPrS2eUTYb5p9QZ8HwwHnW-roARt3SQawaeE6RY1yrWOJrvtjR4lK-l9N5WvNfaf5KLgtGhyEi2GbcfD4S-iQ624drC5GbhbeR9E5c12Rhu9XqL41LB1_E5qZd2ZkdNemq7UIGh4wh0erMK5wI6PH2K0VmPaYjCwdtt7UxjfLTC58q9z-MnEV1ENX3p_KkLR5YvtcA1IlRtLUDzLtgBYQloVghVAjigbreVyrxTs4BDGjqX1GykwvJSr5AK1X0uYC2dHgPklrXiVjnyPupp0K62bgD0YgaLBn1UARW8ykh02hyNVdwF4UmhwL_ZFCAFZScZHB3YkGGhtqb1-h0QVVchMSKnIQShBCokMTJ0iUGI1WL_OlQh50iZDxbwS00Kj-z0py1k2uDMGkGcXvnNRSHca8VdBiRcFVtVaJo0IiyuphLfd88S9b3HDrpVxjHB2ygKtUedpss6CmeX69ccZivrlQbrGj1xLmBfQwIKEzN-9kQbI183Np-NTrSh4iEf2BvcKvRyJ_RgybjvoxgzzwJtfHRS7_Vry8rXvDUhTrovYGp-4MTl47__D0rNwPtCY4EKoU7Io8yecS8K7m7Vld4NxbSwKK5svh74LKstEED_BB59KlpBDmFx32a-VJxIsUASG-0k-7pMHjGizEoghkqO1ebILV0lcDInsqpnE_rPyUq8cXZOECG1uSGunsVM; SID=g.a000owjbfsrZR9lXGf8u5wAe7EuresCxMzaOx61z3ZcKCUGvNSyu4BfzTgWlFJKHeFDWA3uc-AACgYKAXQSARcSFQHGX2MicrCFQtECGgJliRuptvsHnBoVAUF8yKr2mB7ofEvOBDA11mrJrOF90076; __Secure-1PSID=g.a000owjbfsrZR9lXGf8u5wAe7EuresCxMzaOx61z3ZcKCUGvNSyuOow8KoDiOiokF6bPvWuYxAACgYKAaoSARcSFQHGX2MimoprCqRNdoEJ8_61j20WiRoVAUF8yKq2RpkbzOe5M3JC8MdFdyPU0076; __Secure-3PSID=g.a000owjbfsrZR9lXGf8u5wAe7EuresCxMzaOx61z3ZcKCUGvNSyu5VPZH4SSyHaGhXuhsPLLVgACgYKARoSARcSFQHGX2MitKmoBA14sCtB_cnOezvCdhoVAUF8yKoI27wOPRvIy5bWYFfU2dDC0076; HSID=ARsqbV3Xr5jDcoKU3; SSID=AYiy87A23pesRSDEI; APISID=Q0qb1GEbo-rPsx5p/A-UledllC_xlN6FMK; SAPISID=C4I81QwYnnmUoTkv/AVweK7R_NHX3NqB6K; __Secure-1PAPISID=C4I81QwYnnmUoTkv/AVweK7R_NHX3NqB6K; __Secure-3PAPISID=C4I81QwYnnmUoTkv/AVweK7R_NHX3NqB6K; SIDCC=AKEyXzVy0lpXktGiRiJ5e9LASSoJl1IjnGbHlhITe3l9q_ETtCA8hm6rE-BpQnOGjnaphIKduRda; __Secure-1PSIDCC=AKEyXzWP24qJZlrH0xoKdE7z1evgND45AAnJ3fpIo-LnLVAHDyxZps-S5QcPA0mSyQ_NfgeLptWf; __Secure-3PSIDCC=AKEyXzXUuanevqVP64f7sSpcrIS8RYXLp55aoDZmJDiZDEwwh2YFgNx2NXfli1LaI7pVN6ZAHy_x; __Secure-1PSIDTS=sidts-CjEBQT4rX4g_WSIwFx9oSNqnNb9Dy85Ub-Y7yRCNlEFjxgN7Jj9MTM55XwIXelGVlsa3EAA; __Secure-3PSIDTS=sidts-CjEBQT4rX4g_WSIwFx9oSNqnNb9Dy85Ub-Y7yRCNlEFjxgN7Jj9MTM55XwIXelGVlsa3EAA' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: cross-site' -H 'Sec-Fetch-User: ?1' -H 'Priority: u=0, i' -o image.tar
```

```shell
docker load -i image.tar
```

#### Запуск контейнеров проекта
```shell
docker compose -f docker-compose.yml up --build
```

#### Остановка контейнеров проекта
```shell
docker compose -f docker-compose.yml down
```

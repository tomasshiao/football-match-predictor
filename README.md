# Football Match Predictor

This project uses Machine Learning algorithms to generate the most likely scores for a match between two nations.

## Build

### Apple Containers

```sh
make mac-up
make mac-down
```

### Docker

> [!IMPORTANT]
> Check to have enough memory.
> ```sh
> docker info --format 'Memory={{.MemTotal}} CPUs={{.CPU}}'
> ```
> To assign more memory:
> ```sh
> colima stop
> colima start --memory 8 --cpu 4
> ```
> At least 8 GiB is needed to process all the data.

```sh
docker compose up
docker compose down
```

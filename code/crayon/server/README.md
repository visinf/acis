# The server is build completely inside a docker image.

IF you do not have the docker image yet:
```bash
# Get it from Docker Hub
docker pull alband/crayon

# OR build it locally with:
docker build -t alband/crayon:latest .
```

Then the server can be started with:
```bash
docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon [frontend_refresh] [backend_refresh]
```
Where `frontend_refresh` is an optional argument that allows to change the refresh rate of tensorboard to the given value (in seconds). Default is 10 seconds.
Where `backend_refresh` in an optional argument that allows to change how often the tensorflow backend reload the files from disk (in seconds). Default is 0.5 seconds. Reducing this value too much may make tensorboard unstable.


Tensorboard can then be accessed from a browser at `localhost:8888`.
The client should be setup to send the datas at `localhost:8889`.

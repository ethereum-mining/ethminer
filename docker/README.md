# Dockerfile for cpp-ethereum
Dockerfile to build a bleeding edge cpp-ethereum docker image from source

    docker build -t cppeth < Dockerfile

Run a simple peer server

    docker run -i cppeth -m off -o peer -x 256

GUI is compiled but not exposed. You can mount /cpp-ethereum/build to access binaries:

    cid = $(docker run -i -v /cpp-ethereum/build cppeth -m off -o peer -x 256)
    docker inspect $cid # <-- Find volume path in JSON output

You may also modify the Docker image to run the GUI and expose a
ssh/VNC server in order to tunnel an X11 or VNC session.


FROM ubuntu:14.04

ARG PY_VERSION
ARG WHEEL_BASE_URL

# Set up fake user / sudo environment:

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo lsb-release

RUN adduser --gecos "Test User" --disabled-password testuser && \
    test -d /home/testuser

COPY visudo-helper.sh prep-run-test.sh /root/

RUN VISUAL=/root/visudo-helper.sh visudo

COPY test_wrapper.sh /home/testuser
COPY BinaryDrop.tar.gz /home/testuser
RUN chown -R testuser:testuser /home/testuser

# TODO run repeated
RUN su - testuser -c "./test_wrapper.sh BinaryDrop.tar.gz $PY_VERSION $WHEEL_BASE_URL"
RUN /root/prep-run-test.sh

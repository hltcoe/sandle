FROM nginx:1.22
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y npm
COPY nginx.default.conf /etc/nginx/conf.d/default.conf
COPY . /tmp/app-source
RUN cd /tmp/app-source && \
    npm install && \
    npm run build && \
    mv dist/* /usr/share/nginx/html/ && \
    cd / && \
    rm -rf /tmp/app-source
## Clean up
#RUN apt-get purge -y npm && \
#    apt-get autoremove -y

# Ngingx setting up

Check DNS https://www.digitalocean.com/community/tools/dns?domain=artswiper.uk

[https://dnschecker.org/](https://dnschecker.org/)

Domain setup


https://docs.hetzner.com/dns-console/dns/general/authoritative-name-servers/


To configure Nginx so that `greenbuddy.online` points to port `3000` and `api.greenbuddy.online` points to port `8000`, you need to create two separate server blocks in Nginx. Here's how you can do it:

1. **Remove Existing Symlinks:**
    
    First, remove any existing symlinks in `/etc/nginx/sites-enabled/`:
    
    ```bash
    sudo rm /etc/nginx/sites-enabled/*
    
    ```
    
2. **Create or Update the Site Configurations:**
    
    Create or update the configuration files for `greenbuddy.online` and `api.greenbuddy.online` in `/etc/nginx/sites-available/`:
    
    ```bash
    sudo nano /etc/nginx/sites-available/greenbuddy.online
    sudo nano /etc/nginx/sites-available/api.greenbuddy.online
    
    ```
    
    For `greenbuddy.online`, use the following configuration:
    
    ```
    server {
        listen 80;
        server_name greenbuddy.online;
        return 301 https://$host$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name greenbuddy.online;
    
        ssl_certificate /etc/letsencrypt/live/greenbuddy.online/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/greenbuddy.online/privkey.pem;
        include /etc/letsencrypt/options-ssl-nginx.conf;
        ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
        location / {
            proxy_pass <http://0.0.0.0:3000>;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
    
    ```
    
    For `api.greenbuddy.online`, use the following configuration:
    
    ```
    server {
        listen 80;
        server_name api.greenbuddy.online;
        return 301 https://$host$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name api.greenbuddy.online;
    
        ssl_certificate /etc/letsencrypt/live/api.greenbuddy.online/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/api.greenbuddy.online/privkey.pem;
        include /etc/letsencrypt/options-ssl-nginx.conf;
        ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
        location / {
            proxy_pass <http://0.0.0.0:8000>;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
    
    ```
    
    Save and close each file (`Ctrl+O`, `Enter`, `Ctrl+X`).
    
3. **Create the Symlinks:**
    
    Create symbolic links from the sites-available directory to the sites-enabled directory:
    
    ```bash
    sudo ln -s /etc/nginx/sites-available/greenbuddy.online /etc/nginx/sites-enabled/
    sudo ln -s /etc/nginx/sites-available/api.greenbuddy.online /etc/nginx/sites-enabled/
    
    ```
    
4. **Test and Reload Nginx:**
    
    Test the Nginx configuration to ensure there are no syntax errors:
    
    ```bash
    sudo nginx -t
    
    ```
    
    If the test is successful, reload Nginx:
    
    ```bash
    sudo systemctl reload nginx
    
    ```
    

This setup should configure `greenbuddy.online` to proxy requests to port `3000` and `api.greenbuddy.online` to proxy requests to port `8000`, both using HTTPS. If you encounter any errors, please provide the specific error messages for further assistance.
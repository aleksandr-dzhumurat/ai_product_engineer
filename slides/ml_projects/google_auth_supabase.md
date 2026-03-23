# Google auth setup

### Supabase

enage google auth: 

Authentification (left panel) 

→ **Sign In / Providers:** enable **Google auth**

- copy `Client Secret (for OAuth` and `Callback URL`

→ URL configuration

- **Site URL**
- **Redirect URLs**

## Configure Google OAuth Console

Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Credentials → OAuth 2.0 Client IDs:

→ **Authorized JavaScript origin**

→ **Authorized redirect URIs :** place here callback you got from supabse URL

Copy client secret for react app

Step 1: Access Supabase Dashboard

1. Go to [https://app.supabase.com](https://app.supabase.com/)
2. Sign in and select your project: [https://hpzemqnaoaocvizsncnk.supabase.co](https://hpzemqnaoaocvizsncnk.supabase.co/)

Step 2: Enable Google OAuth

1. In your Supabase dashboard, go to Authentication → Providers
2. Find Google in the list of providers
3. Toggle it ON (enabled)

Step 3: Configure Google OAuth Credentials

You'll need to set up Google OAuth credentials:

1. Create Google OAuth App:
- Go to [https://console.cloud.google.com](https://console.cloud.google.com/)
- Create a new project or select existing one
- Enable Google+ API
- Go to Credentials → Create Credentials → OAuth 2.0 Client IDs
- Application type: Web application
- Name: "Webhook Dialog"
2. Configure Redirect URIs:
- Authorized JavaScript origins: [http://localhost:9999](http://localhost:9999/)
- Authorized redirect URIs: [https://hpzemqnaoaocvizsncnk.supabase.co/auth/v1/callback](https://hpzemqnaoaocvizsncnk.supabase.co/auth/v1/callback)
3. Add Credentials to Supabase:
- Copy the Client ID and Client Secret from Google
- In Supabase → Authentication → Providers → Google:
    - Paste Client ID
    - Paste Client Secret
    - Site URL: [http://localhost:9999](http://localhost:9999/)
    - Redirect URLs: [http://localhost:9999](http://localhost:9999/)

Step 4: Save Configuration

Click Save in the Supabase dashboard.

After completing these steps, your Google sign-in should work. The error will disappear and
users will be able to authenticate with Google.

Would you like me to help with any specific part of this setup, or do you need assistance
with creating the Google OAuth credentials?

Deploy container

1. Deploy your container:

# 

# Check if it's running

docker ps
docker logs webhook-dialog

1. Update Supabase & Google OAuth:

Supabase Dashboard:

- Go to [https://app.supabase.com](https://app.supabase.com/) → Authentication → URL Configuration
- Site URL: [http://34.139.81.112](http://34.139.81.112/)
- Redirect URLs: [http://34.139.81.112](http://34.139.81.112/)

Google OAuth:

- Go to [https://console.cloud.google.com](https://console.cloud.google.com/) → APIs & Services → Credentials
- Edit your OAuth 2.0 Client ID
- Authorized JavaScript origins: [http://34.139.81.112](http://34.139.81.112/)
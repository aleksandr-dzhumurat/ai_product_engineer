# Google Cloud Store: data and csv

How to switch-on and use for data analysis

Create bucket in [Google Cloud Console](https://console.cloud.google.com/storage/browser) (like an in you own operating system)

- Creating new account
    
    ![Navigate to Service Accounts in IAM & Admin](../img/gcs_nav_to_service_accounts.png)
    

Then go to to “Service accounts”, find ACC that you just created, Click on it and go to “Keys” tab to create key file in JSON format

- create key
    
    ![Service Account Keys configuration](../img/gcs_service_account_keys.png)
    

You can find description of this process in the [medium article](https://medium.com/google-cloud/automating-google-cloud-storage-management-with-python-92ba64ec8ea8)

I uploaded [movielens 100k](https://grouplens.org/datasets/movielens/100k/) dataset.

Return to [buckets](https://console.cloud.google.com/storage/browser) and upload file in [interface](https://console.cloud.google.com/storage/browser/geo-recommendations-store)

- result
    
    ![Bucket contents with uploaded datasets and models](../img/gcs_bucket_contents.png)
    

Whole code [here](https://github.com/aleksandr-dzhumurat/gcs-workshop/blob/main/src/prepare.py)
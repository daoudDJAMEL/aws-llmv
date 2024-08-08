import os
import time
import boto3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
LOCAL_DIRECTORY = 'D:\\aws-data\\upload'
S3_BUCKET_NAME = 'dataset-api-etats-room'
AWS_ACCESS_KEY_ID = 'AKIA6ODU7NSLF23BDHVH'
AWS_SECRET_ACCESS_KEY = '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'
REPOSITORY_NAME = 'images_upload'  # Add the repository name here

# Initialize S3 client with credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

class S3UploadHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self.upload_to_s3(event.src_path)

    def upload_to_s3(self, file_path):
        try:
            file_name = os.path.basename(file_path)
            s3_path = f"{REPOSITORY_NAME}/{file_name}"  # Include the repository name in the S3 path
            s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_path)
            print(f'Successfully uploaded {file_name} to {S3_BUCKET_NAME}/{REPOSITORY_NAME}')
        except Exception as e:
            print(f'Failed to upload {file_path}: {e}')

if __name__ == "__main__":
    event_handler = S3UploadHandler()
    observer = Observer()
    observer.schedule(event_handler, LOCAL_DIRECTORY, recursive=False)
    observer.start()
    print(f'Started monitoring {LOCAL_DIRECTORY} for new files.')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

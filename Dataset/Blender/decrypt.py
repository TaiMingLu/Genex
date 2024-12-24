from cryptography.fernet import Fernet
import os
from tqdm import tqdm

# Read the key from omnigibson.key
with open('F:/DATASETS/OmniGibson/omnigibson.key', 'rb') as key_file:
    key = key_file.read()

# Initialize the Fernet object with the key
fernet = Fernet(key)


data_path = 'F:/DATASETS/OmniGibson/dataset/objects'
objects = os.listdir(data_path)

for object in tqdm(objects):
    objects_path = os.path.join(data_path, object)
    for object_id in os.listdir(objects_path):
        encrypted_object_path = os.path.join(objects_path, object_id, 'usd', f'{object_id}.encrypted.usd')
        decrypted_object_path = os.path.join(objects_path, object_id, 'usd', f'{object_id}.usd')


        # Read the encrypted file
        with open(encrypted_object_path, 'rb') as enc_file:
            encrypted_data = enc_file.read()

        # Decrypt the data
        decrypted_data = fernet.decrypt(encrypted_data)

        # Write the decrypted data to a new file
        with open(decrypted_object_path, 'wb') as dec_file:
            dec_file.write(decrypted_data)

        # print("Decryption successful, file saved as decrypted_file.usd")

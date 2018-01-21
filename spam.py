# %%
import numpy as np  # you probably don't need this line
from glob import glob
import email

def fetch_emails(path):
  fnames = glob(path)
  emails = []
  for f in fnames:
    file = open(f, 'r',errors='ignore') 
    emails.append(file.read())
  return np.array(emails)

ham = fetch_emails('datasets/ham/*')
email.message_from_string(ham[1])
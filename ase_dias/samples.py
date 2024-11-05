"""
Download sample trajectory files
"""
import os
import requests


def available_data():
  """
  Show available Data Files

  Usage
  -----
  >>> from aimDIAS import available_data()
  >>> available_data()
  """
  samples_url = "https://raw.githubusercontent.com/kangmg/aimDIAS/main/examples/samples.txt"
  response = requests.get(samples_url)
  if response.status_code == 200:
    print(response.text)
  else:
    print(f" Failed to load the available data list.\n Please visit manually. \n URL : https://raw.githubusercontent.com/kangmg/aimDIAS/main/examples/samples.txt")


def load_data(dataName:str, save_file:bool=False, save_folder:str="./"):
  """
  Description
  -----------
  Load sample data files from a github repository. 

  Parameters
  ----------
    - dataName(str): Name of the data to load.
      - You can find all available `Data Name` by executing `available_data()`.
    - save_file(bool): Choose whether to save the data as a file or return it as a string. Default is False. (returns a string)
    - save_folder(str): Folder to save the sample data. Default is the current directory.

  Returns
  -------
    - sample data(str): if `save_file=False`
    - None: if `save_file=True`

  Examples
  --------
  ### Load sample data without saving file
  >>> data = load_data('example.txt')
  
  ### Load sample data and save it as a file in current directory
  >>> load_data('example.txt', save_file=True)
  
  ### Load sample data and save it in a specific directory
  >>> load_data('example.txt', save_file=True, save_folder='/path/to/folder/')
  """
  # load sample data from git repository
  data_url = f"https://raw.githubusercontent.com/kangmg/aimDIAS/main/examples/{dataName}"
  tmp = "https://raw.githubusercontent.com/kangmg/ascii_art/main/ascii_arts/ilovecat"
  
  if dataName == "mycat":
    tmp_response = requests.get(tmp)
    print(tmp_response.text) if tmp_response.status_code == 200 else None
    return None

  # response status code / 200 -> success
  response = requests.get(data_url)
  status_ = response.status_code
  if status_ != 200:
    print(f" Failed to download the {dataName} file.\n Please download manually. \n URL : {data_url}")
    return None
  
  # save sample data
  elif (status_ == 200) and save_file:
    save_path = os.path.join(save_folder, dataName)
    with open(f"{save_path}", 'w') as file:
      file.write(response.text)
    print(f"{dataName} file saved in {save_path}")
    return None
 
  # return data string without file save
  elif (status_ == 200) and (not save_file):
    return response.text

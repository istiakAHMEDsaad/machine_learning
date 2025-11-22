# This method with Tensorflow with Cuda:

1. Download the supported python version [python 3.7.x - 3.10.x](https://www.python.org/downloads/windows/)
2. Download CUDA Toolkit 11.2 [download](https://developer.nvidia.com/cuda-11.2.0-download-archive)
3. Download cuDNN 8.1 [download](https://developer.nvidia.com/rdp/cudnn-archive)</br>
   Or specific link: [February 26th, 2021 CUDA 11.0, 11.1 and 11.2](<https://developer.nvidia.com/rdp/cudnn-archive#:~:text=Download%20cuDNN%20v8.1.1%20(Feburary%2026th%2C%202021)%2C%20for%20CUDA%2011.0%2C11.1%20and%2011.2>)
4. Install CUDA Toolkit 11.2 (.exe file)
5. Extract cuDNN 8.1 then copy (bin, include, lib) from cudnn file then paste it on `c:\Program File\Nvidia GPU Computing Toolkit\CUDA\11.2\`
6. Specific tensorflow version 2.10(last version for windows native gpu) if you need latest version you need wsl or Linux OS `pip install tensorflow==2.10`

# How to setup environment?

- Install python (3.9, 3.10) recommended
- Create a folder then open terminal (windows "cmd or poweshell") (mac or linux default terminal)
- for windows(cmd/powershell): `python -m venv venv` for (linux/mac) `python3 -m venv venv`
- for windows(cmd/powershell): `venv\Scripts\activate` for (linux/mac) `source venv/bin/activate`
- after activate the local environment

- **Install TensorFlow 2.10 GPU FIRST**
  - `pip install tensorflow==2.10 keras==2.10`
- **Install compatible NumPy, SciPy, h5py**
  - `pip install numpy==1.23.5 scipy==1.9.3 h5py==3.8.0`
- **Install JupyterLab + Data Science libs**
  - `pip install jupyterlab pandas==2.0.3 matplotlib seaborn scikit-learn==1.2.2`
- **Install PyTorch (compatible with numpy 1.23.5)**
  - `pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1`
- **Install OpenCV**
  - `pip install opencv-python==4.6.0.66`
- **LOCK NumPy to prevent future upgrades**
  - `pip install "numpy<1.24" --no-cache-dir`
- if it's download a lots of file then please uninstall that version
  ```
  pip uninstall numpy scipy h5py -y
  pip install numpy==1.23.5 scipy==1.9.3 h5py==3.8.0
  # lock the numpy important
  pip install "numpy<1.24" --no-cache-dir
  ```

- **Test TensorFlow GPU**
  ```
  import tensorflow as tf
  print(tf.__version__)
  print(tf.config.list_physical_devices("GPU"))
  ```


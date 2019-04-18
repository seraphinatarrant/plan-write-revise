# Plan-Write-Revise

This repository contains code for the [Plan, Write, Revise](https://arxiv.org/abs/1904.02357) paper. 
Please cite the work if you use it.
`@inproceedings{GoldfarbTarrant2019PlanWA,
  title={Plan, Write, and Revise: an Interactive System for Open-Domain Story Generation},
  author={Seraphina Goldfarb-Tarrant and Haining Feng and Nanyun Peng},
  year={2019}
}`

A working version of this same code and demo can be found at [here](http://cwc-story.isi.edu/)
as used in the paper.

If you have comments/complaints/ideas, please feel free to send me an [email](mailto:serif@uw.edu) or open an issue. 

## Steps to setup

### Download Models
You will need to download and unzip the pretrained models 
[here](https://drive.google.com/drive/folders/19D2uRIaISghKSybiQiVs7uOXxYtAKCMR?usp=sharing). This
contains the language model for the Storyline Planner, the baseline Title-to-Story language model,
the Plan-and-Write language model, and the 2 discriminators , and the matching vocabulary files.
Make sure they are in the `models/` directory, with no nested folders.

### Install Requirements
Use your favorite virtualenv manager and and install `requirements.txt` (in the root dir of this repo).
Note that if you use a package manager like Anaconda, at the time of writing not all requirements
are supported, so you will have to `pip install` whatever is missing into the conda/miniconda/etc 
environment.

Tokenization of user-input (as described in the paper) requires a SpaCy model. 
In the active virtual env, do `python -m spacy download en_core_web_lg`

Out-of-Vocabulary handling is done via WordNet (also in paper). WordNet is most easily downloaded in 
a python shell. If you include the flag `--download-nltk` the first time you start 
the system (via `web_server.py`) it will download the resource as it boots. You only have to do this once
and may remove the flag on subsequent runs. 

### Other System requirements
The system runs on CPU on a Unix-based system. It runs much faster on GPU 
(primarily due to the decoding of the discriminators) but is
designed to run on CPU for maximum portability. The code is easy to GPU enable via a flag and 
we will expose that in this repo if there is interest.

## Run the system
cd into the `server/` directory, and then run the system via `python web_server.py`. If you want to 
specify the particular port, you can do `CWC_SERVER_PORT=[NUM] ./web_server.py` or 
`python web_server.py --port [NUM]`. They are equivalent. More options are available by reading the 
comments under `main` at the bottom of `web_server.py`.

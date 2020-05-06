# MI2T_EE380L
This is a course project of EE380L, Data mining, UT Austin, 2020 Spring semester.


Classifier part:


Musical timbre translation part:
This work is inspired from:
https://github.com/facebookresearch/music-translation
https://arxiv.org/abs/1805.07848

The codes we modify come from:
https://github.com/scpark20/universal-music-translation
The "music_trans.ipynb" is the original code from this URL.


The final code we use is music_trans_8k.py, the music generation code we use is data_processing_8k.py.



Environment:
GPU >= GTX 1080， 8GB RAM
Dataset: IRMAS
* File Name format: X_Y_Z1_Z2.wav.png => X folder (0: part1, 1: part2, 2: part3), Yth wav file, Z: instruments (pia, vio, ...)
* part1 - https://drive.google.com/open?id=1Nj_jdkSZWXni1WETrUZ5XuJzoNZDTeW3
* part3 - https://drive.google.com/open?id=1erkFTdDQjpOfpjb1_O6zF2Wf6CJshbkD


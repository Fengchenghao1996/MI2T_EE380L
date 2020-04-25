# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:41:34 2020

@author: 14124
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:33:01 2020

@author: 14124
"""

src_instrument_index = 0
dest_instrument_index = 1
index = np.random.randint(0, inst_waves_list[src_instrument_index].shape[0], 1)
_src = inst_waves_list[src_instrument_index][index]

_latents = sess.run(up_latents, feed_dict={x_holder: _src})

print(_latents.shape)

plt.figure(figsize=[18, 2])
plt.plot(_src[0])
plt.show()

plt.figure(figsize=[18, 2])
plt.plot(_latents[0])
plt.show()

from tqdm import tqdm
from IPython.display import clear_output

_samples = np.zeros([1, 1024])
_latents = np.concatenate([np.zeros([1, 1024, LATENT_DIM]), _latents], axis=1)
for i in tqdm(range(T)):
    _inference_sample_list = sess.run(inference_sample_list, feed_dict={x_holder: _samples[:, -1024:], 
                                                                        latents_holder: _latents[:, i:i + 1024]})
    _samples = np.concatenate([_samples, np.expand_dims(_inference_sample_list[dest_instrument_index], axis=0)], axis=-1)
    if i % 200 == 0:
        clear_output()
        plt.plot(_src[0, :i])
        plt.show()
        
        plt.plot(_samples[0, 1024:])
        plt.show()

print(_samples.shape)


sr_original=8000
sr_tanslate=8000
librosa.output.write_wav('music_trans_8k/src_1.wav', _src[0], sr_original)
librosa.output.write_wav('music_trans_8k/samples_1.wav', _samples[0], sr_tanslate)

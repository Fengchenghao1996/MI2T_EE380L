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
def m_trans(src, sample):
    src_instrument_index = src
    dest_instrument_index = sample
    index = np.random.randint(0, inst_waves_list[src_instrument_index].shape[0], 1)
    _src = inst_waves_list[src_instrument_index][index]
    
    _latents = sess.run(up_latents, feed_dict={x_holder: _src})
    
    print(_latents.shape)
    
    plt.figure(figsize=[9, 1])
    plt.plot(_src[0])
    plt.show()
    
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
    
    plt.figure(figsize=[18, 10])  
    
    plt.plot(_src[0],label='original')
    plt.plot(_samples[0, 1024:],label='translated')
    plt.xlabel('time/1e-4s')
    plt.ylabel('amplitude')
    plt.show()
  
    plt.savefig('music_trans_bach/results_' + str(src)+'to'+str(sample) + '.png')
    print(_samples.shape)
    src_index = src_instrument_index
    dest_index = dest_instrument_index
    sr_original=8000
    sr_tanslate=8000
    librosa.output.write_wav('music_trans_bach/src_'+str(src_index)+'to'+str(dest_index)+'.wav', _src[0], sr_original)
    librosa.output.write_wav('music_trans_bach/sample_'+str(src_index)+'to'+str(dest_index)+'.wav', _samples[0], sr_tanslate)
    return True

m_trans(0,1)
m_trans(1,0)
m_trans(0,2) 
m_trans(2,0)
m_trans(1,2)
m_trans(2,1)
# m_trans(3,1)
# m_trans(1,3)
# m_trans(3,2) 
# m_trans(2,3)

m_trans(0,0)
m_trans(1,1)
#m_trans(2,2) 
# m_trans(3,3) 

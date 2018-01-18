import os
import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import math
import re
import hashlib

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def seq_to_times(s):
    # Converts a sequence of classes (0/1) for 32ms overlapping windows to a list of times (in seconds) where the class changes
    assert(len(s)>0)
    times = []
    times.append(0.)
    start_class = s[0]
    cur_class = start_class
    cur_time = 0
    for i in s:
        if i != cur_class:
            time_sec = float(cur_time/1000)
            times.append(time_sec)
            cur_class = i
        cur_time = cur_time+16
    times.append(float(cur_time/1000))
    return times

def low_pass_gaussian(s, sigma=1.5):
    return np.round(scipy.ndimage.filters.gaussian_filter1d(s.astype(float), sigma)).astype(int)

def seq_to_mbss_times(s, min_nb=10, tmin=3., tmax = 5.):
    # Simple heuristic to choose the segment(s) on which we want to call mbss.
    # ret[0][0] / ret[0][1] > starting/ending time for the 'speaking' segment
    # ret[1][0] / ret[1][1] > starting/ending time for the 'not speaking' segment

    assert(len(s)>0)
    ret = [[0.,3.],[3.,6.]]
    times = []
    times.append(0.)
    start_class = s[0]
    cur_class = start_class
    cur_time = 0
    # Look for consecutive ones
    cur_ones = 0
    for i in s:
        if i != cur_class:
            time_sec = float(cur_time/1000)
            times.append(time_sec)
            cur_class = i
            if cur_ones>=min_nb:
                pred_time_sec = times[len(times)-2]
                if  time_sec - pred_time_sec <= tmax:
                    if  time_sec - pred_time_sec >= tmin:
                        ret[1] = [pred_time_sec, time_sec]        
                    else:
                        len_seq = float((len(s)-1)*16/1000)
                        ret[1] = [pred_time_sec, min(pred_time_sec+tmax,len_seq)]        
                else:
                    ret[1] = [pred_time_sec, pred_time_sec+tmax]        
                if len(times)>2:
                    pred2_time_sec = times[len(times)-3]
                    if pred_time_sec - pred2_time_sec <= tmax:
                        if pred_time_sec - pred2_time_sec >= tmin:
                            ret[0] = [pred2_time_sec, pred_time_sec]
                        else:
                            ret[0] = [max(0,pred_time_sec-tmin), pred_time_sec]
                    else:
                        ret[0] = [max(0., pred_time_sec-tmax), pred_time_sec]
                else:
                    print('Warning: using beginning of file for noise localization.')
                    ret[0] = [0, tmax]
                break
            cur_ones = 0
        if cur_class == 1:
            cur_ones = cur_ones + 1
        cur_time = cur_time+16
    return ret

def rel_pos_to_angles(x,y,z):
    (a,e,r) = cart2sph(x,y,z)
    ag = a*180/math.pi
    eg = e*180/math.pi
    return [ag, eg]

def cart2sph(t):
    x = t[0]
    y = t[1]
    z = t[2]
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z

def get_speaker_angles(fname):
    # Return the 1x2 array of the true angles of the speaker for audio file [fname].

    home = get_attr(fname, 'home') 
    room = get_attr(fname, 'room') 
    speaker_pos = get_attr(fname, 'speakerPos') 
    array_pos = get_attr(fname, 'arrayPos') 
    # sp_pos_fname = fname.replace('audio','transcriptions')
    sp_pos_fname = 'data/annotations/rooms/home'+home+'_room'+ \
                    room+'_speakerPos'+speaker_pos+'.txt'
    ar_pos_fname = 'data/annotations/rooms/home'+home+'_room'+ \
                    room+'_arrayPos'+array_pos+'.txt'
    sp_pos = np.loadtxt(sp_pos_fname, usecols=(1,2,3,4,5),delimiter='\t')
    ar_pos = np.loadtxt(ar_pos_fname, usecols=(1,2,3,4,5),delimiter='\t')
    diff_pos = sp_pos[0:3] - ar_pos[0:3]
    # print(diff_pos)
    (t, p, r) = cart2sph(diff_pos)
    t = t*180/math.pi
    p = p*180/math.pi
    (t_ar, p_ar) = ar_pos[3], ar_pos[4]
    print('Cube rotation: '+str(t_ar))
    # t_true = t-t_ar
    print('Cube phi: '+str(p_ar))
    # p_true = p-p_ar
    t_true = t
    p_true = p
    res = np.array([t_true,p_true])
    res = res.reshape(1,2)
    return res

def get_noises_angles(fname):
    # Return the nx2 array of the true angles of the noises for audio file [fname],
    # n being the number of noises sources in the annotation.

    res = np.empty((0,2))
    home = get_attr(fname, 'home') 
    room = get_attr(fname, 'room') 
    noise_cond_pos = get_attr(fname, 'noiseCond') 
    if int(noise_cond_pos) != 1:
        # print('Noise_cond is '+str(noise_cond_pos))
        # print(type(noise_cond_pos))
        array_pos = get_attr(fname, 'arrayPos') 
        # sp_pos_fname = fname.replace('audio','transcriptions')
        nc_pos_fname = 'data/annotations/rooms/home'+home+'_room'+ \
                        room+'_noiseCond'+noise_cond_pos+'.txt'
        ar_pos_fname = 'data/annotations/rooms/home'+home+'_room'+ \
                        room+'_arrayPos'+array_pos+'.txt'
        # print(nc_pos_fname)
        nc_pos = np.loadtxt(nc_pos_fname, usecols=(1,2,3),delimiter='\t')
        ar_pos = np.loadtxt(ar_pos_fname, usecols=(1,2,3,4,5),delimiter='\t')
        if len(nc_pos.shape) == 1:
            nc_pos = nc_pos.reshape((1,3))
        nb_src = nc_pos.shape[0]
        res = np.empty((nb_src,2))
        for i in range(nb_src):
            diff_pos = nc_pos[i,0:3] - ar_pos[0:3]
            (t, p, r) = cart2sph(diff_pos)
            t = t*180/math.pi
            p = p*180/math.pi
            # (t_ar, p_ar) = ar_pos[3], ar_pos[4]
            res[i,0] = t
            res[i,1] = p
    return res

def get_attr(s, pattern):
    pat = '(?<='+pattern+')((\d)+)(?=(_|.))'
    res = re.search(pat, s)
    return res.group(0)

def get_speaker(s, pattern):
    pat = '(?<='+pattern+')(M|F)((\d)+)(?=(_|.))'
    res = re.search(pat, s)
    return (res.group(1), res.group(2))

def predict_from_direction(direction_predicted, method = "naive", **kwargs):
    # Very simple heuristics for the association.
    # Sure we can do better than that!
    if method == "naive":
        # print(direction_predicted)
        # print(type(direction_predicted))        
        direction_predicted_class = np.ones(direction_predicted.shape[0], dtype = int)
        direction_predicted_class[0] = 0
        return (direction_predicted_class, direction_predicted)
    elif method == "naive-2calc":
        (dirs_sp, dirs_no) = direction_predicted
        ret1 = dirs_sp.shape[0]
        ret2 = dirs_no.shape[0]
        # Keep first speaking source, and noise sources
        # if ret1 == 0:
        # else:
        dirs_to_keep = np.vstack([dirs_sp[0,:], dirs_no])
        classes = np.ones(dirs_to_keep.shape[0], dtype = int)
        classes[0] = 0
        return (classes, dirs_to_keep)
    elif method == "naive-2calc-combine-1st":
        (dirs_sp, dirs_no) = direction_predicted
        # Keep first speaking source, and noise sources
        ret1 = dirs_sp.shape[0]
        ret2 = dirs_no.shape[0]
        tab_res = []
        if ret1 == 0:
            tab_res.append(dirs_no)
            classes = np.ones(ret2, dtype = int)
            if ret2>0:
                classes[0]=0
            # if len(tab_res)>0:
            dirs_to_keep = np.vstack(tab_res)
            # else:
                # dirs_to_keep = np.empty((0,2))
        else:
            tab_res.append(dirs_sp[0,:])
            if ret1>1:
                common_dirs = find_similar(dirs_sp[1:ret1-1,:], dirs_no)
                if common_dirs.shape[0]>0:
                    tab_res.append(common_dirs)
            else:
                tab_res.append(dirs_no)
            if len(tab_res)>0:
                dirs_to_keep = np.vstack(tab_res)
            else:
                dirs_to_keep = np.empty((0,2))
            classes = np.ones(dirs_to_keep.shape[0], dtype = int)
            classes[0] = 0
        return (classes, dirs_to_keep)
    else:
        raise NameError('Unkwnokn method')

def find_similar(ar1, ar2, eps=10):
    # Find rows of [ar2] that are similar to at least one row of [ar1] with dist <= [eps].
    ret = []
    for i in range(ar1.shape[0]):
        vi = ar1[i,:]
        for j in range(ar2.shape[0]):
            vj = ar2[j,:]
            if np.linalg.norm(vi-vj) <= eps:
                ret.append(vj)
    if len(ret)>0:
        res = np.vstack(ret)
    else:
        res = np.empty((0,2))
    return  res

def found_sources(direction_all_true, dirs_all, epsilon = 10, **kwargs):
    found_sources = []
    for i in range(direction_all_true.shape[0]):
        found = 1
        for j in range(dirs_all.shape[0]):
            if np.linalg.norm(direction_all_true[i]-dirs_all[j])<=epsilon:
                found = 0
                break
        found_sources.append(found)
    return np.array(found_sources)

def associate_classes(direction_all_true, direction_classes_true, direction_predicted, direction_predicted_classes, epsilon = 10.0):
    direction_associate_classes = []
    #print(direction_all_true)
    #print(direction_classes_true)
    #print(direction_predicted)
    #print(direction_predicted_classes)
    for i in range(direction_all_true.shape[0]):
        found = False
        for j in range(direction_predicted.shape[0]):
            if direction_predicted_classes[j] == direction_classes_true[i] and np.linalg.norm(direction_all_true[i]-direction_predicted[j])<= epsilon:
                direction_associate_classes.append(direction_predicted_classes[j])
                found = True
                break
        if not found:
            direction_associate_classes.append(2)
    #print(direction_associate_classes)
    #pause()
    return np.array(direction_associate_classes)

def get_mbss(eng, fname, from_sec, to_sec, nsrc, pooling, use_cache=True):
    # Calculate or load from cache if [use_cache] is True, the mbss directions
    # as a nx2 array extracted on [fname] from [from_sec] to [to_sec] using [nsrc]>=n sources.
    # [pooling] is the pooling method (aggregation over time), [eng] the matlab engine.
    if pooling == 'max':
        str_pooling = ''
    else:
        str_pooling = pooling
    str_cache = fname+('{:.02f}'.format(from_sec))+('{:.02f}'.format(to_sec))+str(nsrc)+str_pooling
    hash = hashlib.md5(str_cache.encode())
    cache_path = 'cache/'+str(hash.hexdigest())+'_mbss.npy'
    if use_cache and os.path.isfile(cache_path):
        print('Loading MBSS from cache.')
        mbss = np.load(cache_path)
    else:
        print('No MBSS cache, calculating.')
        mbss = np.array(eng.mbss_python_wrapper(fname, from_sec, to_sec, nsrc, pooling, nargout=1))
        np.save(cache_path, mbss)
    print('MBSS has size: '+str(mbss.shape))
    return mbss

"""
Code written by José Garrido Ramas
"""
from nuscenes_utils.nuscenes import NuScenes
import numpy as np
from helpers import *
import pickle

#nusc =  NuScenes(version='v0.1', dataroot='/home/jose/data', verbose=True)
with open('../data/nusc.p', 'rb') as pickle_file:
    nusc = pickle.load(pickle_file)

map_data_split = np.load('../data/map_data_split14.npy')

map_data_all = np.load('../data/map_data_all.npy')

def find_closest(point, points):
	point_rep = np.matlib.repmat(point, points.shape[1], 1).T
	dists = np.sqrt(np.sum(np.square((point_rep - points)),0))
	i = np.argmin(dists) 
	return i, dists[i]

def get_lines(scene_ix, delta):

    scenes = nusc.scene
    record = scenes[scene_ix]
    partition = map_data_split[scene_ix][0]
    # instance_tokens = self.map_data_all[scene_ix]['instance_tokens'] #extra index is not useful
    scene_token = record['token']
    no_samples = record['nbr_samples']
    log_record = nusc.get('log', record['log_token'])
    map_record = nusc.get('map', log_record['map_token'])

    # map_record['mask'].mask holds a MapMask instance that we need below.
    map_mask = map_record['mask']

    #Find the limits of the scene
    xmin = 1000000
    xmax = 0
    ymin = 1000000
    ymax = 0 
    for i in range(partition.shape[1]):
        indexes = np.where(partition[:,i,0] != -999)[0]
        for instance_index in indexes:
            x = partition[instance_index][i][0]
            y = partition[instance_index][i][1]
            xmin = min(xmin,x) 
            xmax = max(xmax,x)
            ymin = min(ymin,y)
            ymax = max(ymax,y)

    A= (ymax-ymin)*(xmax-xmin)

    #If area is already big enough, do not expand the map
    if(A>600*600):
        delta = 0
    do = np.array(map_mask.mask[(int(ymin)-delta):(int(ymax)+delta),(int(xmin)-delta):(int(xmax)+delta)])
    mask = Image.fromarray(do)
    docol =np.abs(np.concatenate((np.zeros((do.shape[0],1)),  np.diff(do,axis = 1)), axis = 1))
    dorow = np.abs(np.concatenate((np.diff(do,axis = 0),np.zeros((1,do.shape[1]))), axis = 0))
    edges = docol + dorow
    rows, col = np.where(edges != 0)
    #divide line 
    threshold = 10
    #Find closest point in a list of points
    i = 0
    points_org = []
    traj = []
    trajs = []
    d = 0
    points = np.concatenate((np.expand_dims(rows,0),np.expand_dims(col,0)), axis = 0)
    points.shape
    iters = 0
    stop = points.shape[1]
    while(points.shape[1]>1):
        d = 0 
        #iters+= 1
        if(iters > stop):
	        print('use break')
	        break
        while(d<threshold and points.shape[1]>1 ):
	        point = points[:,i]
	        points_org.append(point)
	        traj.append(point)
	        points = np.delete(points, i, 1) #delete column i
	        i, d = find_closest(point, points)
	        iters += 1
	        if(iters>stop):
	            print('use break')
	            break

	    #Find the closest point    
        trajs.append(traj)
        traj = []
    return trajs,do

def get_length(traj):
	#Sample trajectory to avoid noise in getting the length
	#the length of the trajectory is not proportional to the number of points (point density varies)

	#step = 5 for it to work well on resampled trajectories
	step = max(int(len(traj)/10),1)
	t = np.arange(0, len(traj), step)
	if(len(traj)>1):
	    delta_x = np.diff(traj[t,0])
	    delta_y = np.diff(traj[t,1])   
	    return (sum(np.sqrt(np.power(delta_x, 2)+np.power(delta_y, 2))))
	else: 
	    return 0

def find_drivable(do, px, py, p1x, p1y, step):
	#find where is the drivable area
	#step is the distance to the road
	pmx, pmy = int((p1x+px)/2), int((p1y+py)/2)
	vecx, vecy = -(p1y-py), (p1x- px)
	power = np.sqrt(vecx**2 + vecy**2)
	vecx = vecx*step /power
	vecy = vecy*step / power
	ctex = int(10 *np.sign(vecx))
	ctey = int(10 *np.sign(vecy))
	delta_ = 20
	do_ = np.concatenate((do, np.zeros((delta_, do.shape[1]))), axis = 0)
	do_ = np.concatenate((np.zeros((delta_, do_.shape[1])), do_), axis = 0)
	do_ = np.concatenate((do_, np.zeros((do_.shape[0], delta_))), axis = 1)
	do_ = np.concatenate((np.zeros((do_.shape[0], delta_)), do_), axis = 1)

	if(do_[pmx + ctex+delta_, pmy+delta_] == 255): #drivable area reached through adding VECX
	    signx = '+'
	    
	else: 
	    signx = '-'
	    
	if(do_[pmx+delta_, (pmy + ctey+delta_)] == 255): #drivable area reached through adding VECY
	    signy = '+'
	else: 
	    signy = '-'
	    
	return signx, signy


def get_directions(px, py, p1x, p1y, step, signx, signy):
	#find where is the drivable area
	#step is the distance to the road
	pmx, pmy = int((p1x+px)/2), int((p1y+py)/2)
	vecx, vecy = -(p1y-py), (p1x- px)
	power = np.sqrt(vecx**2 + vecy**2)
	vecx = vecx*step /power
	vecy = vecy*step / power
	ctex = int(10 *np.sign(vecx))
	ctey = int(10 *np.sign(vecy))

	if(signx == '+'): #drivable area reached through adding VECX
	    poutx = pmx + vecx
	    
	else: 
	    poutx = pmx - vecx
	    
	if(signy == '+'): #drivable area reached through adding VECY
	    pouty = pmy + vecy
	else: 
	    pouty = pmy - vecy
	    
	return poutx, pouty

def get_next_point(pix, traj, step_len):
	#Find the point in trajectory TRAJ that is length STEP_LEN from point (px, py)
	px, py = traj[pix]
	pixf = pix+1

	while(get_length(traj[pix:pixf]) < step_len):
	    pixf += 1
	    
	return pixf
	  
def resample_traj(traj):
	traj_resampled = []
	for i in range(traj.shape[0]-1):
	    length  = get_length(traj[i:(i+2)])
	    vx, vy = traj[i+1,0]-traj[i,0], traj[i+1,1]-traj[i,1]
	    #normalize vx, vy
	    cte = 2/ np.sqrt(vx**2 +vy**2)
	    vx = cte * vx
	    vy = cte * vy
	    px, py = traj[i,0], traj[i,1]
	    traj_add = [[px,py]]
	    traj_resampled.append([px, py])
	    while(get_length(np.array(traj_add))<length):
		    px += vx
		    py += vy
		    traj_add.append([px, py])
		    traj_resampled.append([px,py])
	return np.array(traj_resampled)

def get_new_traj(traj_resampled, velocities, margin_error = 50):
	#Margin_error is necessary because the measurement of the length is not exact. 
	#Imagine a new trajectory from a resampled one. points = length(velocities)+ 1

	#length of the new trajectory
	L_new = sum(velocities)
	L_old = get_length(traj_resampled)

	#in this case we need to extend the old trajectory

	if(L_old < L_new+margin_error):
	    traj_resampled = extend_traj(traj_resampled, L_new +margin_error)
	    

	L_old = get_length(traj_resampled)
	px, py = traj_resampled[0]
	j = 0
	traj_new  =[[px,py]]
	for v in velocities:
	    px,py,j = get_next_point(j, traj_resampled, v)
	    traj_new.append([px, py])

	return np.array(traj_new)
		    
def get_next_point(ixp, traj, step_pos):
	#Find the point in trajectory TRAJ that is length STEP_LEN from point (px, py)
	px, py = traj[ixp]
	ixpf = ixp+1
	#print('Length of the trajectory: ', get_length(traj))
	#print('we look for a point at distance: ', step_pos)
	while(get_length(traj[ixp:ixpf]) < step_pos and (ixpf<traj.shape[0])):
	    ixpf += 1
	if(ixpf < traj.shape[0]):
	    px, py = [traj[ixpf,0], traj[ixpf,1]]
	else:
	    print('error')
	    print(traj)
	    px, py = -999,-999

	return px,py, ixpf


def get_new_trajs(trajs_div,do,test, show = False):
    mask = Image.fromarray(do)
    if(show):
        _, axes = plt.subplots(1, 1, figsize=(10, 10))
        plt.ion()
        axes.imshow(mask.resize((int(mask.size[0]), int(mask.size[1])),
		                resample=Image.NEAREST))

    train_scenes = list(range(90))
    test_scenes = list(range(90,100))
    velocities_train = build_velocities(train_scenes)
    velocities_test = build_velocities(test_scenes)
    if(test):
        velocities_ = velocities_test
    else:
        velocities_ = velocities_train

    trajs_new = []
    for i in range(len(trajs_div)):
        step_t = 15

        traj = trajs_div[i]
        t = np.arange(0, traj.shape[0], step_t)

        der = np.diff(traj[t,:], 2, axis = 0)
        dermax = np.sum(np.max(np.abs(der), axis =  0))

        if(dermax<20):
            if(show):
                axes.plot(traj[t,1], traj[t,0], color = 'b')
            #Now create a face traj
            traj_sim = np.zeros((10,2))
            traj_sim = []
            step_pos = 50
            i_offset = 30
            t = 0
            distance = 30 #max distance to the road of the generated trajectories
            step = np.round(np.random.random() * distance)
            #for t in range(14):
            point_x, point_y = traj[t*step_pos,0], traj[t*step_pos,1]
            point1_x, point1_y = traj[i_offset+t*step_pos,0], traj[i_offset+t*step_pos,1]
            signx, signy = find_drivable(do, point_x, point_y, point1_x, point1_y, step)
            while(t*step_pos+i_offset < len(traj)):
                point_x, point_y = traj[t*step_pos,0], traj[t*step_pos,1]
                point1_x, point1_y = traj[i_offset+t*step_pos,0], traj[i_offset+t*step_pos,1]
                #traj_sim[t,0], traj_sim[t,1] = get_directions(point_x, point_y, point1_x, point1_y)
                px_moved, py_moved =  get_directions(point_x, point_y, point1_x, point1_y, step, signx, signy)
                traj_sim.append([px_moved, py_moved])
                t+=1

            #resample traj_sim
            traj_sim = np.array(traj_sim)
            traj_resampled = resample_traj(traj_sim)
            L_sim = get_length(traj_sim)
            margin_error = 50
            i = np.random.randint(0,len(velocities_))
            vel = velocities_[i]
            while(L_sim  < sum(vel) + margin_error):
                i = np.random.randint(0,len(velocities_))
                vel = velocities_[i]
            
            start = np.random.randint(0,L_sim - np.floor(sum(vel)) - margin_error+1)
            _,_,start_ix = get_next_point(0,traj_resampled, start)
            traj_new = get_new_traj(traj_resampled[start_ix:], vel)
            trajs_new.append(traj_new)
            #create new traj    
            if(show):
                axes.plot(traj_new[:,1], traj_new[:,0], color = 'g', marker = 's',markersize=2)
    return trajs_new
	    
#divide trajs
def divide_trajs(trajs):
	trajs_div = []
	L = 800
	step =  300
	for i in range(len(trajs)):
	    traj = np.array(trajs[i])
	    offset = 0
	    trajs_ = []
	    ix = 0
	    dermax = 30
	    step = 20  
	    #Get the least linear line
	    while((traj.shape[0]-offset)>L):
		    offset += step
		    traj_ = np.array(traj[offset:(offset+L)])
		    t = np.arange(0, traj_.shape[0], step)
		    der = np.diff(traj_[t,:], 2, axis = 0)
		    der = np.sum(np.max(np.abs(der), axis =  0))
		    if(der < dermax):
		        dermin = der
		        ixmin = ix
		        trajs_div.append(traj_)
		    ix +=1

	return trajs_div


def extend_traj(traj,new_length): 
    """
    Extend trajectory TRAJ so it has length NEW_LENGTH. To achieve this we assume the vehicle
    #maintains the velocity (vx and vy) it had during the last 10% of its trajectory
    """
    print('A trajectory is being extended')
    t = np.linspace(0, len(traj)-1, split_size, dtype = int)
    L = get_length(traj)
    vx, vy = np.diff(traj[t], axis = 0)[split_size -2]

    epsilon = 10**(-12) #numerical stability purposes
    cte = 2/ np.sqrt(vx**2+vy**2+epsilon)

    vx = vx *cte
    vy= vy *cte
    if(np.abs(vx)<0.1 and np.abs(vy) <0.1):
        print('error, too small a final velocity, object is stopped in the end.')
        print(traj)
        print(vx, vy)
        vx, vy = traj[len(traj)-1] - traj[0]

    px, py = traj[len(traj)-1,0], traj[len(traj)-1,1]
    traj_add = [[px,py]]
    while(get_length(np.array(traj_add))<new_length-L):
        px += vx
        py += vy
        traj_add.append([px, py])      
        
    new_traj = np.zeros((len(traj_add)+traj.shape[0],2))
    new_traj[0:traj.shape[0]] = traj
    new_traj[traj.shape[0]:] = np.array(traj_add)

    return new_traj
	    
def build_velocities(train_scenes):
	#Build velocities matrix
	THRESHOLD_ACCELERATION = 7 #MAXIMUN  ACCELERATION CONSIDERED NORMAL
	THRESHOLD_LENGTH = 40
	#Let's look at the number of partitions per instance
	#Problem with this approach: same trajectory will appear a lot of times
	velocities = []
	instance_ID = 0
	ins_offset = 0
	part_no = {}

	map_data_split_ = map_data_split[train_scenes] #to ensure we train with only velocities that come from the training set
	for scene_ix in range(map_data_split_.shape[0]):
	    traj_partitions = []
	    for partition_ix in range(len(map_data_split_[scene_ix])):
		    part = map_data_split_[scene_ix][partition_ix]
		    traj_instances= []
	        
		    for instance_ix in range(part.shape[0]):
		        instance_ID = ins_offset +instance_ix
		        traj_x = map_data_split_[scene_ix][partition_ix][instance_ix, :, 0]
		        traj_y = map_data_split_[scene_ix][partition_ix][instance_ix, :, 1]
		        traj = np.array([traj_x, traj_y])
		        if instance_ID not in part_no: 
		            part_no[instance_ID]= 0
		            
		        if(~np.any(traj_x == -999) and ~np.any(traj_y == -999)):
		            part_no[instance_ID]+=1
		            vx = np.diff(traj_x)
		            vy = np.diff(traj_y)
		            v = np.sqrt(np.power(vx, 2)+np.power(vy,2))
		            length = get_length(traj)
		            a = np.abs(np.diff(traj, 2, axis = 1))
		            a_ = np.sqrt(np.power(a[:,0],2) + np.power(a[:,1],2))
		            L_v = sum(v)
		            if(L_v> THRESHOLD_LENGTH and np.max(a_)<THRESHOLD_ACCELERATION):
		                velocities.append(v)
	    ins_offset = instance_ID+1
	    
	return np.array(velocities)

def load_maps():
    maps_list = []
    #Save map mask for every scene
    for i in range(100):
        scenes = nusc.scene
        record = scenes[i]
        scene_token = record['token']
        no_samples = record['nbr_samples']
        log_record = nusc.get('log', record['log_token'])
        map_record = nusc.get('map', log_record['map_token'])
        #map_record['mask'].mask holds a MapMask instance that we need below.

        map_mask = map_record['mask']
        maps_list.append(map_mask.mask)
        print('Finished loading the map corresponding to scenee: ', i)

    return maps_list



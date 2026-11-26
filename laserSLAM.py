import os
from module1 import *
from loopdetection import *
from posegraph import *
from math import pi
import matplotlib.pyplot as plt

lidar, lidar_data, name = ReadData(1)
N = lidar_data.shape[0]

# Output directories
os.makedirs(name + "/results", exist_ok=True)
os.makedirs("results", exist_ok=True)

#Map parameters
borderSize = 1
pixelSize = 0.2
miniUpdated = False
miniUpdateDT = 0.1
miniUpdateDR = 5*pi/180

resolution = [0.1,0.1,pi/180]
fastResolution = [0.05,0.05,0.5*pi/180]
bruteResolution = [0.01,0.01,0.05*pi/180]
tempResolution = [i/2 for i in fastResolution]

#Create an empty map
gmap = Map()
graph = PoseGraph(verbose = True)
infm = np.array([[100.0, 0.0, 0.0],[0.0, 100.0, 0.0],[0.0, 0.0, 100000.0]])
loop_infm = np.array([[200.0, 0.0, 0.0],[0.0, 200.0, 0.0],[0.0, 0.0,200000.0]])


#Start
pose = [0,0,0]
path = [pose]
maps = []
scans = []
loop_pairs = []

vfile = open(name+"/vfile.txt", 'w')
efile = open(name+"/efile.txt", 'w')

for scan_id in range(0,N,2):

    if scan_id%100==0:
        print('scan '+str(scan_id))
    
    scan = lidar.ReadAScan(lidar_data, scan_id, 60)
    scans.append(scan)
    maps.append(Transform(scan,pose))
    
    #Initialize
    if scan_id == 0:
        gmap.initialize(pose, scan)
        graph.addNode(scan_id, pose, vfile)
        miniUpdated = True
        count = 0
        continue
        
    #Match current scan to local map#
    count += 1

    #update local points map and local grid map
    if miniUpdated:
        localMap = gmap.ExtractLocalMap(pose, scan, borderSize)
        gridMap1 = OccuGrid(localMap, pixelSize)
        gridMap2 = OccuGrid(localMap, pixelSize/2)
    
    #Predict current pose using constant velocity motion model
    if scan_id > 20:
        pose_guess = [pose[i] + DiffPose(path[-2],pose)[i] for i in range(3)]
    else:
        pose_guess = pose

    #Fast Matching
    if miniUpdated:
        [pose, _] = FastMatch(gridMap1, scan, pose_guess, fastResolution)
    else:
        [pose, _] = FastMatch(gridMap2, scan, pose_guess, fastResolution)
        
    
    #Refine the pose using smaller pixels
    pose, hits = FastMatch(gridMap2, scan, pose, tempResolution)


    #If the robot moved far enough since the last keyscan, add scan points to the map
    dp = DiffPose(gmap.keyscans[-1].pose,pose)
    dp = [abs(x) for x in dp]
    if dp[0]>miniUpdateDT or dp[1]>miniUpdateDT or dp[2]>miniUpdateDR:
        miniUpdated = True
        pose = gmap.Add_a_keyscan(gridMap2, scan, pose, hits, pixelSize, bruteResolution, 0.1, 3*pi/180)
    else:
        miniUpdated = False

    path.append(pose)

    #Add graph node
    graph.addNode(count, pose, vfile)

    #Add graph edge
    T = getTransform(path[-2],pose)
    graph.addEdge([[count-1, count, T]], infm, efile)


    #Loop Closing
    if miniUpdated:
        if TryLoopOrNot(gmap):
            gmap.keyscans[-1].loopTried = True
            fit_idxs = getLoopCandidate(path, maps,fix=100,max_dist = 3, e_threshold = 3, nb_mode =2)
            if fit_idxs != None:
                gmap.keyscans[-1].loopClosed = True
                for i in range(len(fit_idxs)):
                    idx = fit_idxs[i]
                    if [idx, scan_id] not in graph.edges_list:
                        if i in range(2) or i in range(len(fit_idxs)-1,len(fit_idxs)):
                            gridmap = OccuGrid(maps[idx], pixelSize)
                            pose_guess = pose
                            est_pose,_ = FastMatch(gridmap, scan, pose_guess, resolution)
                            est_pose,_ = FastMatch(gridmap, scan, est_pose, fastResolution)
                            est_pose,_ = FastMatch(gridmap, scan, est_pose, tempResolution)
                            dp_ = DiffPose(est_pose, pose)
                            lp=[idx,scan_id, dp_, est_pose, pose, path[idx]]
                            if i in range(2):
                                loop_pairs.append(lp+[0])
                                T= getTransform(path[idx],est_pose)
                                e_list = [[idx,count,T]]
                                graph.addEdge(e_list, loop_infm, efile)
                            elif i == len(fit_idxs)-1:
                                T = getTransform(path[idx],est_pose)
                                e_list=[[idx,count, T]]
                                loop_pairs.append(lp+[1])
                                plt.plot(lp[3][0], lp[3][1], 'x', markersize=10, color = 'blue')
                                plt.scatter(lp[4][0], lp[4][1], s=10, c = 'r')
                                plt.title("Scan: "+str(scan_id))
                                _path = name + "/results/scan_"+str(scan_id)+".png"
                                plt.savefig(_path,dpi=300)
                                plt.clf()
                            else:
                                T = getTransform(path[idx],est_pose)
                                e_list=[[idx,count, T]]
                                loop_pairs.append(lp+[1])
                                plt.plot(lp[3][0], lp[3][1], 'x', markersize=10, color = 'blue')

    #Plot
    if scan_id%50 == 0:
        PlotMap(gmap, path, scan, scan_id)

vfile.close()
efile.close()

###########################################
path = np.array(path)
plt.scatter(path.T[0], path.T[1], s=3)
for lp in loop_pairs:
    if lp[6] == 0:
        plt.plot(lp[3][0], lp[3][1], 'x', markersize=10, color = 'green')
    elif lp[6] ==1:
        plt.plot(lp[3][0], lp[3][1], 'x', markersize=10, color = 'blue')
    plt.scatter(lp[4][0], lp[4][1], s=10, c = 'r')
    plt.scatter(lp[5][0], lp[5][1], s=10, c = 'm')

plt.show()
plt.clf()
#################################################


vfile = name + "/vfile.txt"
efile = name + "/efile.txt"

graph = PoseGraph(verbose=True)
graph.readGraph(vfile, efile)
path = graph.optimize(4, plt, name)

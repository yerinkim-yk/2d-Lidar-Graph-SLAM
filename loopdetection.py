from math import cos, sin, pi
from module1 import *
import open3d as o3d
import scipy
from sklearn.neighbors import NearestNeighbors

def TryLoopOrNot(gmap):
    #Do not try loop closing too often (at least 10 keyscans since the last trial)
    trial_interval = 1
    for k in range(len(gmap.keyscans)-1,-1, -1):
        if not gmap.keyscans[k].loopTried:
            trial_interval +=1
        else:
            break
    if trial_interval < 10:
        needLoopTrial = False
        return needLoopTrial

    #Must have traveled far enough since the last closed loop
    traveledDistance = 0
    for k in range(len(gmap.keyscans)-2, -1, -1):
        if not gmap.keyscans[k].loopClosed:
            dT = DiffPose(gmap.keyscans[k].pose, gmap.keyscans[k+1].pose)
            traveledDistance += (dT[0]**2+dT[1]**2)**0.5
            if traveledDistance>5:
                break
        else:
            break
    if traveledDistance<3:
        needLoopTrial = False
        return needLoopTrial

    needLoopTrial = True

    return needLoopTrial
    

def np2pcd(xyz):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("sync.ply")
    return pcd_load


def getLoopCandidate(path, maps, fix=30, max_dist = 1.5, e_threshold = 0.7, nb_mode=1):
    kd = scipy.spatial.cKDTree(np.array(path)[:, :2])
    poseid = len(path) - 1
    pose = path[poseid]
    scan = maps[poseid]
    idxs = kd.query_ball_point(np.array([pose[0], pose[1]]), r=max_dist)
    fit_idxs = []
    errors = []
    # RMSE value is obtained from the Euclidean distance between corresponding point pairs in both point-clouds, used for the ICP registration
    for idx in idxs:
        if idx != poseid and poseid - idx >= fix:
            past_source_points = np.hstack((maps[idx], np.zeros((maps[idx].shape[0],1))))
            past_source = np2pcd(past_source_points)
            target = np2pcd(np.hstack((scan, np.zeros((scan.shape[0],1)))))

            # NN Test
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(np.asarray(target.points))
            distances, indices = neigh.kneighbors(np.asarray(past_source.points), return_distance=True)
            errors.append(np.mean(distances))
            fit_idxs.append(idx)

    loop_candidate = []  # saved pose index

    if len(errors) > 0:
        for n in range(nb_mode):
            _idx = np.argmin(np.array(errors))
            max_fit_idx = fit_idxs[_idx]
            if errors[_idx] <= e_threshold:
                print(errors[_idx])
                loop_candidate.append(max_fit_idx)
            errors[_idx] = 100

        return loop_candidate

    return None

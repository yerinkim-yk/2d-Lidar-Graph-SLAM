import csv
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from math import pi, cos, sin
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors
import scipy
from utils import t2v, v2t

def pol2cart(angles, ranges):
    cart = []
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    cart = np.array([xs,ys]).T    #N*2 array

    return cart

def DiffPose(pose1, pose2):
    dp = [0,0,0]
    dp[0] = pose2[0]-pose1[0]
    dp[1] = pose2[1]-pose1[1]
    dp[2] = pose2[2]-pose1[2]
    dp[2] = (dp[2]+pi)%(2*pi)-pi
    return dp

def getTransform(pose1, pose2):
    node1 = v2t(pose1)
    node2 = v2t(pose2)
    transformation = t2v(np.dot(np.linalg.inv(node1), node2))
    return transformation

def applyTransform(T, pose):
    transformation = v2t(T)
    pose = v2t(pose)
    new_pose = np.dot(pose,transformation)
    return t2v(new_pose)

def Transform(scan, pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    trans = np.array([[c,-s,pose[0]],[s,c,pose[1]],[0,0,1]])
    tscan = np.dot(trans,np.hstack((scan,np.ones((scan.shape[0],1)))).T)[:2].T 
    return tscan

def ReadData(data):
    if data ==0:
        #Deutsches Museum Data
        name = "Deutsches_Museum"
        lidar = Lidar(-2.351831,2.351831,1079)
        mat_file = io.loadmat("data/horizental_lidar.mat")
        lidar_data = np.array(mat_file['ranges'])

    elif data ==1:
        #SNU Library Data
        name = "SNU_Library"
        lidar = Lidar(-pi,pi,898)
        f = open("data/laser.txt", 'r')
        lines = f.readlines()
        ranges = []
        for line in lines:
            ranges += list(map(float, line.split()))
        lidar_data = np.array(ranges).reshape(-1,898)
        f.close()
    elif data ==2:
        name = "full_Library"
        lidar = Lidar(-pi,pi,721)
        f = open("data/lidar_2.txt",'r')
        lines = f.readlines()
        ranges = []
        for line in lines:
            if '[' in line:
                line = line[1:]
            elif ']' in line:
                line = line[:-2]
            ranges += list(map(float, line.split()))
        lidar_data = np.array(ranges).reshape(-1,721)
    else:
        #Simulation Data
        name = "Simulation"
        lidar = Lidar(-pi,pi,360)
        f = open("data/range.csv", 'r')
        rdr = csv.reader(f)
        ranges = []
        for line in rdr:
            ranges += list(map(float,line))
        lidar_data = np.array(ranges).reshape(-1,360)
        f.close()
    return lidar, lidar_data, name

class Lidar():
    def __init__(self, angle_min, angle_max, npoints, range_min = 0.23, range_max = 60):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = (angle_max-angle_min)/npoints
        self.npoints = npoints
        self.range_min = range_min
        self.range_max = range_max
        self.scan_time = 0.025
        self.time_increment = 1.736112e-05
        self.angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

    def ReadAScan(self,lidar_data, scan_id, usableRange):
        ranges = lidar_data[scan_id]

        #Remove points whose range is not so trustworthy
        maxRange = min(self.range_max, usableRange)
        angle = self.angles[(self.range_min<ranges) & (ranges<maxRange)]
        range = ranges[(self.range_min<ranges) & (ranges<maxRange)]

        #Convert from polar coordinates to cartesian coordinates
        scan = pol2cart(angle,range)

        return scan

class keyscan():
    def __init__(self, pose, iBegin, iEnd, loopClosed = False, loopTried = False):
        self.pose = pose
        self.iBegin = iBegin
        self.iEnd = iEnd
        self.loopClosed = loopClosed
        self.loopTried = loopTried

class Map():
    def __init__(self):
        self.points = []
        self.keyscans = []

    def initialize(self, pose, scan):
        self.points = Transform(scan, pose)
        self.keyscans.append(keyscan(pose, 1, scan.shape[0], True))

    def ExtractLocalMap(self, pose, scan, borderSize):
        #Transform current scan to world frame
        scan_w = Transform(scan, pose)

        #Set top-left and bottom-right corner
        minX = min(scan_w.T[0])-borderSize
        minY = min(scan_w.T[1])-borderSize
        maxX = max(scan_w.T[0])+borderSize
        maxY = max(scan_w.T[1])+borderSize

        #Extract
        x = self.points.T[0]
        y = self.points.T[1]
        idx = np.where((minX<x)&(x<maxX)& (minY<y)&(y<maxY))
        local_points = self.points[idx]
        return np.array(local_points)    #M*2 array

    def Add_a_keyscan(self, gridmap, scan, pose, hits, pixelSize, Resolution, tmax, rmax):

        #Evaluate the pose
        last_keypose = self.keyscans[-1].pose
        dp = DiffPose(last_keypose, pose)

        #Add only the points whose hit value is greater than 1.1
        scan_w = Transform(scan, pose)
        new_points = scan_w[np.where(hits>1.1)]
        if new_points.size == 0:
            return pose

        self.keyscans.append(keyscan(pose, self.points.shape[0], self.points.shape[0]+new_points.shape[0]))
        self.points = np.append(self.points, new_points, axis=0)

        return pose


class Gridmap():
    def __init__(self, occGrid, pixelSize, minXY):
        self.occGrid = occGrid
        bwdist =distance_transform_edt(np.ones(occGrid.shape)-occGrid)
        self.metricMap = np.minimum(bwdist,10*np.ones(bwdist.shape))
        self.pixelSize = pixelSize
        self.topLeftCorner = minXY

def OccuGrid(pts, pixelSize):

    #grid size
    minXY = np.min(pts,axis=0) - 3 *pixelSize   #1*2
    maxXY = np.max(pts,axis=0) + 3 *pixelSize   #1*2
    Sgrid = np.flip(np.around((maxXY-minXY)/pixelSize).astype(int)) + np.array([1,1])  #1*2

    grid = np.zeros(Sgrid)
    ix = np.around((pts.T[0]-minXY[0])/pixelSize).astype(int)
    iy = np.around((pts.T[1]-minXY[1])/pixelSize).astype(int)
    idx = np.vstack((iy,ix))
    grid[idx[0],idx[1]] = 1
    gmap = Gridmap(grid, pixelSize, minXY)

    return gmap

def FastMatch(gridMap, scan, pose, searchResolution):

    ipixel = 1/gridMap.pixelSize
    minX = gridMap.topLeftCorner[0]
    minY = gridMap.topLeftCorner[1]
    nCols = gridMap.metricMap.shape[1]
    nRows = gridMap.metricMap.shape[0]

    #Go down the hill
    maxIter = 50
    maxDepth = 3
    iter = 0
    depth = 0

    pixelScan = scan * ipixel
    bestPose = pose
    bestScore = float('inf')

    t = searchResolution[0]
    r = searchResolution[2]

    while iter<maxIter:
        no_Change = True

        #Rotation
        for theta in [pose[2]-r, pose[2], pose[2]+r]:
            ct = cos(theta)
            st = sin(theta)
            S = np.dot(np.array([[ct,-st],[st,ct]]),pixelScan.T).T #N*2

            #Translation
            for tx in [pose[0]-t, pose[0], pose[0]+t]:
                Sx = np.around(S.T[0]+(tx-minX)*ipixel)+1
                for ty in [pose[1]-t, pose[1], pose[1]+t]:
                    Sy = np.around(S.T[1]+(ty-minY)*ipixel)+1

                    i = np.where((1<Sx)&(Sx<nCols)&(1<Sy)&(Sy<nRows))
                    idx = np.around([Sy[i]-1,Sx[i]-1]).astype(int)
                    hits=(gridMap.metricMap[idx[0],idx[1]])
                    score = hits.sum()
                    
                    #update
                    if score < bestScore:
                        no_Change = False
                        bestPose = [tx, ty, theta]
                        bestScore = score
                        bestHits = hits

        if no_Change:   #No improvement in this iteration -> refine the resolution and try again
            r = r/2
            t = t/2
            depth += 1
            if depth > maxDepth:
                break

        #Update the pose and repeat
        pose = bestPose
        iter += 1

    return bestPose, bestHits



def PlotMap(gmap, path, scan,scan_id):
    world = np.array(gmap.points)
    scan = Transform(scan, path[-1])
    path = np.array(path)

    plt.scatter(world.T[0],world.T[1], s = 0.1, c = 'm')
    plt.scatter(scan.T[0], scan.T[1], s=0.2, c = 'b')
    plt.scatter(path.T[0], path.T[1], s=0.5, c='g')
    plt.axis([-22,22,-22,22])
    plt.gca().set_aspect('equal')
    plt.title("Scan: "+str(scan_id))

    _path = "results/scan_"+str(scan_id)+".png"

    plt.savefig(_path,dpi=300)
    plt.clf()

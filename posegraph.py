from utils import t2v, v2t, readPoseGraph
from solvers import *

import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv
import time

class Edge(object):
    def __init__(self, id_from = None, id_to = None, mean = None, infm = None):
        self.id_from = id_from # viewing frame of this edge
        self.id_to = id_to # pose being observed from the viewing frame
        self.mean = mean.flatten() if type(mean) == np.ndarray else mean # Predicted virtual measurement
        self.infm = infm # Information matrix of this edge

class PoseGraph(object):
    #POSEGRAPH A class for doing pose graph optimization
    
    def __init__(self, nodes=None, edges=[], verbose=False):
        # Constructor of PoseGraph
        self.nodes = [] if nodes is None else nodes # Pose nodes in graph. Each row has 3 values: x,y,yaw
        self.edges = [] # Edges in graph
        self.H = [] # Information matrix
        self.b = [] # Information vector
        # if x contains correct poses, then H*x = b
        self.edges_list = []
        self.verbose = verbose
        if len(edges)>0: 
            for e in edges:
                self.edges.append(Edge(*e))

    def addNode(self, idx, pose, vfile= None):
        """
        (1) Function
            - Add node object to graph
        (2) Input
            - idx  : index of node
            - pose : pose information of node
        """
        self.nodes.append(pose)
        if vfile != None:
            vfile.write("VERTEX2 "+ str(idx)+" "+ str(pose[0]) + " "+ str(pose[1])+" "+str(pose[2])+"\n")

    def addEdge(self, id_from_id_to_mean, infm, efile = None):
        """
        (1) Function
            - Add factor between two nodes
        (2) Input
            - idx_from : index of the node where the observation was made - current
            - idx_to   : index of the observed node - past
            - mean     : mean of measurement
            - infm     : information matrix of measurement
        """
        for e in id_from_id_to_mean:
            id_from = e[0]
            id_to = e[1]
            mean = e[2]
            edge = Edge(id_from, id_to, mean, infm)
            self.edges.append(edge)
            self.edges_list.append([id_from, id_to])

        if efile != None:
            str_edge = ""
            for e in id_from_id_to_mean:
                id_from = e[0]
                id_to = e[1]
                mean = e[2]
                str_edge = str_edge + str(id_from)+" "+str(id_to)+ " " +str(mean[0,0])+ " " +str(mean[1,0])+ " " +str(mean[2,0]) + " "
            efile.write("EDGE2 " + str_edge +str(infm[0,0])+" "+str(infm[1,0])+" "+str(infm[1,1])+" "+str(infm[2,2])+" "+str(infm[0,2])+" "+str(infm[1,2])+"\n")
    
    def readGraph(self, vfile, efile=None, from_to_order=True, one_line = False):
        # Reads graph from vertex and edge file (g2o format - see https://github.com/RainerKuemmerle/g2o)
        if efile is None:
            self.nodes, constraints = readPoseGraph(vfile)
            self.nodes = np.array(self.nodes)
            for e in constraints:
                self.edges.append(Edge(*e))
        else:
            # vertex file
            vertices = np.loadtxt(vfile, usecols=range(1,5))
            for i in range(vertices.shape[0]):
                self.nodes.append(vertices[i, 1:4])
            self.nodes = np.array(self.nodes, dtype=np.float64)
            
            # edge file
            f = open(efile,'r')
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                edges = line[1:]
                edges = list(map(float, edges))
                k = len(edges)
                infm = np.zeros((3,3), dtype=np.float64)
                # edges[i, 5:11] ... upper-triangular block of the information matrix (inverse cov.matrix) in row-major order

                infm[0,0] = edges[k-6]
                infm[1,0] = infm[0,1] = edges[k-5]
                infm[1,1] = edges[k-4]
                infm[2,2] = edges[ k-3]
                infm[0,2] = infm[2,0] = edges[k-2]
                infm[1,2] = infm[2,1] = edges[k-1]
                j = 0
                while j + 7 < k:
                    if from_to_order:
                        id_from = int(edges[j])
                        id_to = int(edges[j+1])
                    else:
                        id_from = int(edges[j+1])
                        id_to = int(edges[j])
                    mean = edges[j+2:j+5]
                    edge = Edge(id_from,id_to, mean, infm)
                    self.edges.append(edge)
                    j += 5
            f.close()
    
    def plot(self,plt=None, name="optimization", title=''):
        if plt is not None:
            plt.clf()
            plt.scatter(self.nodes[:,0], self.nodes[:,1], s=2)
            plt.title(title)
            time.sleep(0.1)
            plt.draw()
            plt.savefig(str(name)+"/iter="+title)
            plt.show()
    
    def optimize(self, n_iter=1, plt=None, name="optimization"):
        # Pose graph optimization
        self.plot(plt, name, str(0))
        for i_iter in range(n_iter):
            if self.verbose: print('Pose Graph Optimization, Iteration %d.\n' % i_iter)
            
            # Create new H and b matrices each time
            self.H = np.zeros((len(self.nodes)*3,len(self.nodes)*3), dtype=np.float64)   # 3n x 3n square matrix
            self.b = np.zeros((len(self.nodes)*3,1), dtype=np.float64) # 3n x 1  column vector
            
            if self.verbose: print('Linearizing.\n')
            self.linearize()
            
            if self.verbose: print ('Solving.\n')
            self.solve(i_iter)
            
            if plt is not None:
                self.plot(plt,name, str(i_iter+1))

        return self.nodes

    def linearize(self):
        # Linearize error functions and formulate a linear system
        for i_edge in range(len(self.edges)):
            ei = self.edges[i_edge]
            # Get edge information
            i_node = ei.id_from
            j_node = ei.id_to
            T_z = v2t(ei.mean)
            omega = ei.infm
            
            # Get node information
            v_i = self.nodes[i_node]
            v_j = self.nodes[j_node]
            
            T_i = v2t(v_i)
            T_j = v2t(v_j)
            R_i = T_i[:2,:2]
            R_z = T_z[:2,:2]
            
            si = np.sin(v_i[2])
            ci = np.cos(v_i[2])
            dR_i = np.array([[-si, ci], [-ci, -si]], dtype=np.float64).T
            dt_ij = np.array([v_j[:2] - v_i[:2]], dtype=np.float64).T
            
            # Caluclate jacobians
            A = np.vstack((np.hstack((np.dot(-R_z.T,R_i.T), np.dot(np.dot(R_z.T, dR_i.T), dt_ij))), [0, 0, -1]))
            B = np.vstack((np.hstack((np.dot(R_z.T,R_i.T), np.zeros((2,1), dtype=np.float64))), [0, 0, 1]))
            
            # Calculate error vector
            e = t2v(np.dot(np.dot(inv(T_z), inv(T_i)), T_j))
        
            # Formulate blocks
            H_ii =  np.dot(np.dot(A.T , omega), A)
            H_ij =  np.dot(np.dot(A.T , omega), B)
            H_jj =  np.dot(np.dot(B.T , omega), B)
            b_i  = np.dot(np.dot(-A.T , omega), e)
            b_j  = np.dot(np.dot(-B.T , omega), e)
            
            # Update H and b matrix
            # #(3*(id)):(3*(id+1)) converts id to indices in H and b
            self.H[(3*i_node):(3*(i_node+1)),(3*i_node):(3*(i_node+1))] += H_ii
            self.H[(3*i_node):(3*(i_node+1)),(3*j_node):(3*(j_node+1))] += H_ij
            self.H[(3*j_node):(3*(j_node+1)),(3*i_node):(3*(i_node+1))] += H_ij.T
            self.H[(3*j_node):(3*(j_node+1)),(3*j_node):(3*(j_node+1))] += H_jj
            self.b[(3*i_node):(3*(i_node+1))] += b_i
            self.b[(3*j_node):(3*(j_node+1))] += b_j

    def solve(self, i_iter=0):
        # Solves the linear system and update all pose nodes
        if self.verbose: print('Poses: %d, Edges: %d\n', len(self.nodes), len(self.edges))
        # The system (H b) is obtained only from relative constraints.
        # H is not full rank.
        # We solve this by anchoring the position of the 1st vertex
        # This can be expressed by adding teh equation
        # dx(1:3,1) = 0
        # which is equivalent to the following
        self.H[:3,:3] += np.eye(3)
        
        H_sparse = csr_matrix(self.H) # coo_matrix
        
        dx = spsolve(H_sparse, self.b)
        
        dx[:3] = [0,0,0]
        dx[np.isnan(dx)] = 0
        dpose = np.reshape(dx, (len(self.nodes), 3))
        
        self.nodes += dpose # update

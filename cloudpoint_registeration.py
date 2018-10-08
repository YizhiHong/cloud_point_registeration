import numpy as np
from sklearn import neighbors
from pylab import trace,identity,argmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def get_nearest_neighbor(source, target):
    '''
    Find the Euclidean nearest neighbor in target for each point in source
    ref:'https://github.com/ClayFlannigan/icp/blob/master/icp.py'
    Input:
        source: source point cloud
        target: target point cloud
    Output:
        dist: Euclidean distances of the nearest neighbor
        idx: dst indices of the nearest neighbor
    '''

    N = neighbors.NearestNeighbors(n_neighbors=1)

    N.fit(target)

    dist, idx = N.kneighbors(source, return_distance=True)

    return dist.ravel(), idx.ravel()


def get_rotation_matrix(qr):
	'''
    compute the rotation matrix R
    ref: http://www.cs.virginia.edu/~mjh7v/bib/Besl92.pdf
    Input:
        M: The unit quaternion vector
    Output:
        R: Rotation Matrix R 
    '''
	q0,q1,q2,q3 = qr[0],qr[1],qr[2],qr[3]

	R1 = [q0**2+q1**2-q2**2-q3**2,         2*(q1*q2-q0*q3),             2*(q1*q3 + q0*q2)]
	R2 = [2*(q1*q2+q0*q3),         q0**2+q2**2-q1**2-q3**2,             2*(q2*q3 - q0*q1)]
	R3 = [2*(q1*q3-q0*q2),                 2*(q2*q3+q0*q1), 	  q0**2+q3**2-q1**2-q2**2]

	R = np.array([R1,R2,R3])
	return R


def compute_registeration(P0, Yk):
	'''
    compute the registeration. which is transformation martix
    ref: http://www.cs.virginia.edu/~mjh7v/bib/Besl92.pdf
    Input:
        P0: Point set 
        Yk:  P0 corresponding Point set
    Output:
        qk: transform matrix
    '''

    # compute mu p and mu x

	mup = np.mean(P0, axis=0)

	Nx = len(P0)
	mux = np.mean(Yk, axis=0)
	
	# Cross-covariance martix sigma_px

	sigma_px =  (np.dot(P0.T,Yk) - np.dot(mup.T,mux))/Nx

	Aij = sigma_px - sigma_px.T
	tr = trace(sigma_px)

	Sym = sigma_px + sigma_px.T - tr*identity(3)

	Q_sigma_px = np.array([
        [tr,        Aij[1][2],     Aij[2][0],    Aij[0][1]], 
        [Aij[1][2], Sym[0][0],             0,            0],
        [Aij[2][0],         0,     Sym[1][1],            0],
        [Aij[0][1],         0,             0,    Sym[2][2]]])

    # eigenvalue to get optimal R and T

	w, v = np.linalg.eig(Q_sigma_px)

	qR = v[:, argmin(w)]

	R = get_rotation_matrix(qR)
	qT = (mux.T - R.dot(mup.T)).T

	return qR, qT

def apply_registration(qk, P0):
	'''
    apply registration to P0
    Input:
       	P0: Point set 
        qk: transform matrix
    Output:
        registed: registed Matrix
    '''
	R = qk[0]
	T = qk[1]
	R = get_rotation_matrix(R)
	registed = P0.dot(R.T) + T
	return registed


def icp(P0, X, k = 20, tau = 1/100):
	'''
    Iterative Closest Points algorithm
    Matching 3-D pointcloud
    ref: http://www.cs.virginia.edu/~mjh7v/bib/Besl92.pdf
    Input:
       	P0: Point set 
        X: Target Point set
        k: maximum iteration
        tau: stop mse threshold
    Output:
        R: Rotation Matrix 
        T: Translatetion Vector
    '''

    # make points homogeneous
	h = P0.shape[1] 

	src = np.ones((h+1, P0.shape[0]))
	src = np.copy(P0.T)

	dst = np.ones((h+1, X.shape[0]))
	dst = np.copy(X.T)

	prev_error = 0
	pk = None # Start P

	for i in range(k):
		print('--------Iteration:', i+1, '---------')
		# handling the fisrt case
		if pk is not None:
			src = np.ones((h+1, pk.shape[0]))
			src = np.copy(pk.T)

		# Part a: Compute closest points

		print('a.Compute closest points...')
		dist, idx = get_nearest_neighbor(src.T, dst.T)

		# get corresponding points
		Yk = np.array(dst[:h,idx].T)

		# Compute the registration
		print('b.Compute the registration...')

		qR,qT = compute_registeration(P0, Yk)
		qk = (qR,qT)

		# Apply the registration
		print('c.applying registration...')

		pk = apply_registration(qk, P0)

		# Apply the registration
		print('d.computing mean square error...')

		mean_error = np.mean(dist)
		diff = abs(prev_error - mean_error)
		print('error falls: ',diff)

		if prev_error != 0:
			if diff < tau:
				break

		prev_error = mean_error

	R = get_rotation_matrix(qk[0])
	T = qk[1]


	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax = plt.subplot(1, 1, 1, projection='3d')
	# P0 = P0[:50000]
	# X = pk[:50000]
	# ax.scatter(P0[:, 0], P0[:, 1], P0[:, 2], c='r')
	# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='g')

	# re = apply_registration(qk,P0)

	# ax.plot(P0, X)
	# ax.plot3D(P0[0], P0[1], P0[2], 'gray')
	# plt.show()


	return R, T


    # return np.array(get_rotation_matrix(qk[0])), np.array(qk[1]).reshape((3,1))


def read_data():
	
	pf1 = open('./point_cloud_registration/pointcloud1.fuse')

	# preprocesing string to float

	P0 = np.array([[ float(f) for f in line.split(' ')[:-1] ] for line in pf1.readlines()])
	pf1.close()

	pf2 = open('./point_cloud_registration/pointcloud2.fuse')

	# preprocesing string to float

	X = np.array([[ float(f) for f in line.split(' ')[:-1] ] for line in pf2.readlines()])
	pf2.close()

	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax = plt.subplot(1, 1, 1, projection='3d')
	# P0 = P0[:3000]
	# X = X[:5000]
	# ax.scatter(P0[:, 0], P0[:, 1], P0[:, 2], c='r')
	# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='g')
	# plt.show()å


	return P0,X


def main():

	# read pointcloud1 and pointcloud2, preprocessing to float number.
	print('Loading pointcloud1 and pointcloud2...')
	P0, X = read_data()

	print('Processing ICP(Iterative Closest Points) ')
	print('Computing the translation matrix...')

	R, T = icp(P0, X)


	# now you can use R*P+T to transform the point set

	print('Rotation Matrix R = ', R)

	print('Translation Vector T = ', T)

	pass


if __name__ == "__main__":
    main()







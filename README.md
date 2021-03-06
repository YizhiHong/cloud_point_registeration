# How to run
python cloudpoint_registeraton.py

# make sure the path of point_cloud_registration folder and its document right
# you can also change the path from cloudpoint_registeraton.py.


Input:
2 point clouds of the same scene (point_cloud_registration.rar)
Output:
Transformation matrix to align point cloud 1 to point cloud 2

<h3>1.1 Registration - Goal</h3>
1. To transform sets of surface measurements into a common coordinate system
2. A model shape and a data shape

<h3>1.2 Registration</h3>
1. Issue: Finding corresponding points.
2. ICP: Assume closest points correspond to each other, compute the best transform.


<h3>2.The ICP Algorithm</h3>
1. Given point set P and match point X (two 3D point cloud in our case)
2. Setting P0 = P, initialize start registration vector qk (k=0) = [1,0,0,0,0,0,0]t Set up stop
tolerance threshold   and Max iterations(optional) k. dk = 0
3. For 0 to k:
a. Compute closest points: Yk = C(Pk,X)
b. Compute the registration(transformation): (qk,dk) = Q(P0, Yk)
c. Apply the registration: Pk = (P0,qk)
d. Terminate when mean square error falls below  : dk - dk+1 <  
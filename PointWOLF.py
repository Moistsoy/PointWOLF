import torch
import torch.nn as nn
import numpy as np

class PointWOLF(object):

    def __init__(self, args):
        
        # "args" are predetermined by cmd input, through PointWOLF setting parameters in main.py file.

        self.num_anchor = args.w_num_anchor     # by user input (main.py line79)
        self.sample_type = args.w_sample_type   # by user input (main.py line80)
        self.sigma = args.w_sigma               # by user input (main.py line81)

        self.R_range = (-abs(args.w_R_range), abs(args.w_R_range))  # ROTATION range by user input 
        self.S_range = (1., args.w_S_range)                         # SCALING range by user input
        self.T_range = (-abs(args.w_T_range), abs(args.w_T_range))  # TRANSLATION range by user input
        # NEED TO ADD SHEARING RANGE

        self.Sh_range = (-abs(args.w_Sh_range), abs(args.w_Sh_range))


    def __call__(self, pos):
        """
        input:
            pos([N, 3])

        output:
            pos([N, 3]) : original pointcloud
            pos_new([N, 3]) : point cloud augmented by PointWOLF
        """
        M = self.num_anchor # M x 3 ; M개의 anchor(M), anchor마다 x, y, z 좌표(3)
        N, _ = pos.shape    # N ; 원래 point cloud의 점 개수만 가져옴. pos는 N x 3 모양인데 3은 좌표에 해당. 그 중 점 개수인 N을 추출.

        if self.sample_type == 'random':
            idx = np.random.choice(N, M)    # M ; N개의 점들 중에 M개만 고름. index이므로 크기가 M인 vector.
        elif self.sample_type == 'fps':
            idx = self.fps(pos, M)          # M ; 원래 point cloud 점들 중에 M 개를 고름. 첫 점은 random, 나머지는 farthes point로 sampling (FPS)

        pos_anchor = pos[idx]   # M x 3 ; 원래 point cloud인 pos에서 아까 뽑은 M개의 index를 이용해 anchor point 찾고 좌표 정보 정리.

        pos_repeat = np.expand_dims(pos, 0).repeat(M, axis=0)
        # 원래 point cloud의 차원을 row 방향으로 하나 늘림.
        # ex) [0 1] -> [[0 1]]
        # 그리고 늘린 matrix의 원소들을 row 방향으로 M번 반복함.
        # ex) [[0 1][2 3]] -> [[0 1][0 1][2 3][2 3]]
        # M x N x 3 ; pos[N x 3] 이 차원이 늘어나 pos[?? x N x 3]이 되고, 여기서 repeat으로 인해 pos[M x N x 3]이 됨.
        
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype)
        # zeros_like는 input으로 받는 변수와 동일한 size의 0배열을 return함.
        # M x N x 3 ; 단, 값은 전부 0

        # Move to canonical space -> 위상 공간으로 이동. 3차원 점들이 병렬적으로 놓여진 4차원 정도로 생각하면 될 듯.
        pos_normalize = pos_repeat - pos_anchor.reshape(M, -1, 3)
        # anchor 좌표 다 모아둔 matrix(pos_anchor)도 M x ?? x 3으로 차원 변형
        # anchor point에서는 좌표값들이 다 0으로 지정되는 게 가능해지도록 pos_repeat에서 pos_anchor.reshape를 뺌
        # 위 과정을 normalize라고 부른 듯함.
        # 이게 있어야 args.??_range를 통해 parameter조절이 편해짐.
        # Why? anchor point가 가장 단순한 원점 형태이기 때문.

        # Local transformation at anchor point
        pos_transformed = self.local_transformaton(pos_normalize)
        # local_transformation 돌리고 난 결괏값이 pos_transformed

        # Move to origin space
        pos_transformed = pos_transformed + pos_anchor.reshape(M, -1, 3)
        # M x N x 3 ; 아까 Move to canonical space 파트에서 anchor point를 원점 취급 시킬 수 있도록 이동을 했는데,
        # 이 과정을 다시 거꾸로 pos_anchor을 더해줌으로써 공간적으로 복귀시킴.


        # kernel_regression을 거침.
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)
        pos_new = self.normalize(pos_new)
        # Normalize 과정을 거침. Gaussian 같음 (추정)

        return pos.astype('float32'), pos_new.astype('float32')



    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N, 3])
            pos_anchor([M, 3])
            pos_transformed([M, N, 3])
        
        output :
            pos_new([N, 3]) : pointcloud after weighted local transformation
        """
        # pos_transform에서 M, N 추출.
        M, N, _ = pos_transformed.shape

        # Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor, 1).repeat(N, axis=1) - np.expand_dims(pos, 0).repeat(M, axis=0) # (M, N, 3), d
        # anchor point들의 좌표를 모아둔 pos_anchor matrix를 column방향으로 차원 늘림. 그리고 column방향으로 N번 반복
        # M x 3 이었던 anchor point가 M x N x 3됨
        # 거기서 original pointcloud 였던 pos의 차원은 row 방향으로 차원 늘리고 row 방향으로 M 번 반복.
        # N x 3 이었던 point cloud matrix가 M x N x 3됨.
        # 서로 차를 구하니 anchor point와 entire point들의 거리(d)가 나옴.

        # get_random_axis 함수로 아무 축이나 갖고 와서 그걸 project_axis로 삼음.
        project_axis = self.get_random_axis(1)

        # project_axis(1 x 3)를 column 방향으로 차원 늘림. 
        projection = np.expand_dims(project_axis, axis=1)*np.eye(3) # (1, 3, 3)
        #[[1 0 0 ][0 1 0 ][0 0 1]] 행렬 (np.eye(3))과 project_axis의 column을 늘려서 곱함. 
        # 정사영? 같은 게 되는 듯함.

        #Project distance
        sub = sub @ projection # (M,N,3)
        # @는 matmul임. distance와 projection을 곱해 distance를 project함.
        # 아마 distance를 x, y, z 성분으로 나누어 저장할 수 있게 해주는 듯. (정사영)
        
        sub = np.sqrt(((sub) ** 2).sum(2)) #(M,N)  
        # 각각의 성분들을 제곱하고 더하고, root 씌워서 진짜 제대로 된 거리(scalar 값)를 구함. 그래서 M x N 크기의 결과물.
        

        #Kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))  #(M,N) 
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0) #(N,3)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T) # normalize by weight
        # 솔직히 여기는 뭘 하는 건지 잘 모르겠음.

        return pos_new


    
    def fps(self, pos, npoint):
        """
        input : 
            pos([N,3])
            npoint(int)
            
        output : 
            centroids([npoints]) : index list for fps
        """
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_) #(M)
        distance = np.ones(N, dtype=np.float64) * 1e10 #(N)
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    


    def local_transformaton(self, pos_normalize):
        """
        input :
            pos([N,3]) 
            pos_normalize([M,N,3])
            
        output :
            pos_normalize([M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        M,N,_ = pos_normalize.shape
        # 표준화 시킨 pos_normalize에서 M 과 N값 가져옴.

        transformation_dropout = np.random.binomial(1, 0.5, (M,4)) #(M,4)
        # 난수 생성

        transformation_axis =self.get_random_axis(M) #(M,3)
        # 어느 방향으로 transform할 것인지 M개 추출. M = anchor point 개수임.

        degree = np.pi * np.random.uniform(*self.R_range, size=(M,3)) / 180.0 * transformation_dropout[:,0:1]
        # M x 3 ; sampling from (-R_range, R_range) 
        # R_range 범위에서 M x 3 크기의 난수를 균일하게 생성함. 이걸 pi랑 곱하고 180으로 나누어서 호도법으로 바꿈.

        scale = np.random.uniform(*self.S_range, size=(M,3)) * transformation_dropout[:,1:2]
        # M x 3 ; sampling from (1, S_range)
        # S_range 범위에서 M x 3 크기의 난수를 균일하게 생성함.
        scale = scale*transformation_axis
        # 이것과 축을 곱해 정사영시키고,
        scale = scale + 1*(scale==0)
        # 다시 여기다 1을 더함. 만약 S_range에서 0.4를 뽑았다면 0.4 x scale , 3을 뽑았다면 3 x scale이 되었을 듯.
        # Scaling factor must be larger than 1
        
        trl = np.random.uniform(*self.T_range, size=(M,3)) * transformation_dropout[:,2:3]
        # M x 3 ; sampling from (1, T_range)
        # Translation을 위해 T_range안의 난수를 균일하게 생성. 
        trl = trl*transformation_axis
        # 축하고 곱해줌.
        
        # SHEARING
        shear = np.random.uniform(*self.Sh_range, size = (M, 3)) * transformation_dropout[:, 3:4]
        shear = shear * transformation_axis


        # Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3) # scailing factor to diagonal matrix (M,3) -> (M,3,3)
        # 아까 구한 scale의 차원을 column방향으로 하나 늘리고, 이걸 eye(e)이랑 곱해서 크기를 조정하고 성분을 반영.
        # M x 3 -> M x 3 x 3

        # Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)

        # SHEARING MATRIX
        Sh = np.expand_dims(shear, axis=1) * (np.ones((3, 3)) - np.eyes(3))
        Sh = shear + np.eye(3)

        pos_normalize = pos_normalize@R@S@Sh + trl.reshape(M,1,3)
        
        return pos_normalize



    def get_random_axis(self, n_axis):
        """
        input :
            n_axis(int)
            
        output :
            axis([n_axis,3]) : projection axis   
        """
        axis = np.random.randint(1,8, (n_axis)) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz    
        # 축을 고르는 과정을 1~8까지의 숫자를 2진화 시켜 encoding했음.
        
        m = 3 
        # m=3은 축의 개수가 3개(x, y, z)임을 의미함.

        axis = (((axis[:,None] & (1 << np.arange(m)))) > 0).astype(int)
        return axis



    def normalize(self, pos):
        """
        input :
            pos([N,3])
        
        output :
            pos([N,3]) : normalized Pointcloud
        """
        pos = pos - pos.mean(axis=-2, keepdims=True)
        # 두 번쨰로 낮은 차원(axis=-2)을 기준으로 mean을 구함.
        # input에서 그 값을 빼서 편차를 pos라고 잡음.

        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        # 다차원 matrix인 input의 편차의 제곱의 합을 sqrt했으므로 1/sigma 중에 가장 큰 값 계산.
        # 그 값에 0.999999를 곱해서 거의 동일한 scale을 뽑음.
        # 하지만 normalize이므로 1이 되면 애매해서 0.999999 같은 수를 사욯한 듯.
        
        pos = scale * pos
        # 마지막으로 input 값에 scale을 곱해 return함. 
        return pos
        # 전반적으로 Gaussian normalization 같음.
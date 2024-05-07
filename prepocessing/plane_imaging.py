# file for imaging plane data
# 希望是一个全流程的，包含fft，BP以及其他CS方法的仿真，可以写成一个类别，或者函数
import numpy as np
import os
import scipy.io as scio
from prepocessing_utils import *
from CS_Algorithms import *

# 数据预处理类 —— 原始数据从training set获得全采样数据，然后再进行相关成像和提取距离相处理
# 最后数据全部写入了'data/all_plane_data.npy中，后续成像实验不再需要该类
class PlaneHRRPData:
    def __init__(self, 
                 data_root, 
                 plane_list, 
                 ele_list, 
                 ):
        self.data_root  = data_root
        self.plane_list = plane_list
        self.ele_list   = ele_list
        self.full_sampling_hrrp_shape = [3600, 256, 11]
        # variable for store all plane data
        self.plane_data = {}        
        
        # read data
        self._read_all_plane_hrrp_data()
        # double azi range to 0-359
        self._double_azi()
        pass
    
    # 保存数据
    def _save_hrrp_data(self, save_path):
        np.save(save_path, self.plane_data)

    # 不全至360度数据
    def _double_azi(self):
        for i in self.plane_list:
            tmp = self.plane_data[str(i)]
            self.plane_data[str(i)] = np.concatenate((self.plane_data[str(i)], np.flipud(tmp))).transpose(1, 0, 2)
    
    # 读取单文件数据
    def _read_data_from_TrainingSet(self, plane_num, azi):
        # 每一个.mat文件中存储的是原始HRRP序列，0-3599为（0-180度方位的采样）
        mat_path = os.path.join(self.data_root, '{}'.format(plane_num), 'TrainingSet', '{}.mat'.format(azi))
        hrrp = scio.loadmat(mat_path)['hrrp']       # 256*11 (对应11个俯仰角度)
        return hrrp 
    # 读取单机型数据
    def _read_single_plane_hrrp_data(self, plane_num, azi_range, azi_step):
        # initialize single plane data
        plane_num_str = '{}'.format(plane_num)
        self.plane_data[plane_num_str] = np.zeros(self.full_sampling_hrrp_shape) + np.zeros(self.full_sampling_hrrp_shape)*1j
        for azi in range(0, azi_range, azi_step):
            azi_data = self._read_data_from_TrainingSet(plane_num=plane_num, azi=azi)
            # 然后和已经有的进行拼接
            self.plane_data[plane_num_str][azi, :, :] = azi_data
    # 读取所有飞机数据
    def _read_all_plane_hrrp_data(self):
        for i in self.plane_list:
            self._read_single_plane_hrrp_data(plane_num=i, azi_range=3600, azi_step=1)

# 实验的飞机数据 
#todo
class MeasuredPlaneHRRPData:
    def __init__(self, 
                 data_root, 
                 plane_list, 
                 ele_list, 
                 ):
        self.data_root  = data_root
        self.plane_list = plane_list
        self.ele_list   = ele_list
        self.full_sampling_hrrp_shape = [3600, 256, 11]
        # variable for store all plane data
        self.plane_data = {}        
        
        # read data
        self._read_all_plane_hrrp_data()
        # double azi range to 0-359
        self._double_azi()
        pass
    
    # 保存数据
    def _save_hrrp_data(self, save_path):
        np.save(save_path, self.plane_data)



# 成像类: 包含载入保存好的数据以及各类成像方法
class ImagingPlane():
    def __init__(self, 
                 data_path='data/all_plane_data.npy', 
                 ):
        self.plane_data = np.load(data_path, allow_pickle=True).item()

    def _get_data_from_config(self, plane_num=0, ele=0, azi_start=0, azi_end=10, azi_step=1):
        # 需要指定角度起始、角度间隔、角度结束
        # 将数组平移3600点，正对飞机的角度放在距离像最中间
        roll_plane_data = np.roll(self.plane_data[str(plane_num)], shift=3600, axis=1)
        data = roll_plane_data[:, azi_start:azi_end:azi_step, ele]
        return data
        
    # imaging specified plane
    def _imaging_specified_plane(self, 
                                 plane_num=0, 
                                 ele=0, 
                                 azi_start=0, 
                                 azi_end=10, 
                                 azi_step=1, 
                                 method="PD"
                                 ):
        data = self._get_data_from_config(plane_num=plane_num, ele=ele, azi_start=azi_start, azi_end=azi_end, azi_step=azi_step)
        self._imaging(data=data, method=method)

    # imaging是一个独立的模块，直接接受输入数据，然后进行成像
    def _imaging(self, data, method="PD", is_vis=True): 
        if method == "PD": 
            img = self._PD_imaging(data, is_vis)
            pass
        elif method == "BP":
            img = self._BP_imaging(data, is_vis)
            pass
        elif method == "PR":
            img = self._PolarRefmt_imaging(data, is_vis)
            pass
        elif method == "OMP":
            img = self._OMP_imaging(data, is_vis)
            pass
        elif method == "ADMM":
            img = self._ADMM_imaging(data, is_vis)
            pass
        elif method == "ADMM-Net":
            # 后续准备练一个ADMM Net！然后就可以直接load保存好的模型，方便他们水论文
            pass
        else:
            raise NotImplementedError

        return img

    def _PD_imaging(self, data, is_vis=True):
        # data shape
        (hrrp_points, phi_points) = data.shape
        # 根据data shape确定成像大小
        fft_sz = phi_points
        # fft imaging
        img = np.fft.fftshift(np.fft.fft(data, n=fft_sz, axis=1), axes=1)
        # visualize
        plot_img(data=img, is_db=False, is_map=True)
        return img
    
    def _PolarRefmt_imaging(self, data, is_vis=True):
        # reformating的度数要控制在一个合理的范围内
        # data 应该给出角度间隔来计算
        ## 系统参数（后续可以参数化）
        (hrrp_points, N_phi) = data.shape
        fc = 8.5e9
        c = 299792458
        f_min = 8e9
        f_max = 9e9
        
        phic = 0*np.pi/180; # center of azimuth look angles
        # phi_min = -7.5*np.pi/180; # lowest angle
        # phi_max = 7.5*np.pi/180; # highest angle
        phi_min = -(N_phi/2 * 0.05) * np.pi / 180
        phi_max = (N_phi/2 * 0.05) * np.pi / 180
        
        nSampling = 128             # 距离维度采样点数
        nSampling_angle = N_phi       # 数据采样点数长度？
        
        # 带宽(距离维度)
        f = np.linspace(f_min, f_max-(f_max - f_min)/(nSampling), nSampling)
        k = 2 * np.pi * f / c
        k_max = np.max(k)
        k_min = np.min(k)
        kc = (k_max + k_min) / 2        # 中心频率（波数域）
        # 角度(跨距离维度)
        phi = np.linspace(phi_min, phi_max-(phi_max - phi_min)/(nSampling_angle), nSampling_angle)
        
        # 原始网格
        kx = np.outer(k, np.cos(phi))
        ky = np.outer(k, np.sin(phi))
        
        kx_max = np.max(kx)
        kx_min = np.min(kx)
        ky_max = np.max(ky)
        ky_min = np.min(ky)
        
        MM = 1  # 上采样系数
        kx_step = (kx_max - kx_min)/(MM*(nSampling + 1) - 1)
        ky_step = (ky_max - ky_min) / (MM * (nSampling_angle + 1) - 1)    # 这里可以用来定义成像大小
        
        kx = np.arange(kx_min, kx_max + kx_step ,kx_step)
        ky = np.arange(ky_min, ky_max + ky_step, ky_step)
        Nx = len(kx)
        Ny = len(ky)
        # kx[MM * (nSampling + 1)] = 0            # 对一些异常值做修正
        # ky[MM * (nSampling_angle + 1)] = 0
        kx = np.append(kx, 0)
        ky = np.append(ky, 0)

        # 从hrrp获取k域数据
        # Es = np.fft.fft( np.fft.fftshift(data, 1), axis=0)
        Es = np.fft.fft( data, n=hrrp_points, axis=0 )      # numpy.fft中的傅里叶变换谱就是从-fs/2~fs/2，不需要做fftshift
        Es = Es[:128, :]        # 因为产生的是单边谱，只取其中一半。这一步如果原始回波就处理成频域，是不需要的
        newEs = np.zeros((MM * (nSampling_angle + 1) + 1, MM * (nSampling + 1) + 1)) + 1j * np.zeros((MM * (nSampling_angle + 1) + 1, MM * (nSampling + 1) + 1))        # 新插值后的矩阵
        
        for t, tmpk in enumerate(k):
            for v, tmpPhi in enumerate(phi):
                tmpkx = tmpk * np.cos(tmpPhi)
                tmpky = tmpk * np.sin(tmpPhi)
                indexX = int(np.floor((tmpkx - kx_min) / kx_step))
                indexY = int(np.floor((tmpky - ky_min) / ky_step))

                r1 = np.sqrt(abs(kx[indexX] - tmpkx)**2 + abs(ky[indexY] - tmpky)**2)
                r2 = np.sqrt(abs(kx[indexX+1] - tmpkx)**2 + abs(ky[indexY] - tmpky)**2)
                r3 = np.sqrt(abs(kx[indexX] - tmpkx)**2 + abs(ky[indexY+1] - tmpky)**2)
                r4 = np.sqrt(abs(kx[indexX+1] - tmpkx)**2 + abs(ky[indexY+1] - tmpky)**2)

                R = 1/r1 + 1/r2 + 1/r3 + 1/r4

                A1 = Es[t, v] / (r1 * R)
                A2 = Es[t, v] / (r2 * R)
                A3 = Es[t, v] / (r3 * R)
                A4 = Es[t, v] / (r4 * R)

                newEs[indexY, indexX] += A1
                newEs[indexY, indexX+1] += A2
                newEs[indexY+1, indexX] += A3
                newEs[indexY+1, indexX+1] += A4
        # Down sample newEs by MM times
        # newEs = newEs[::MM, ::MM]   
        
        # 二维fft成像
        # img = np.fft.fftshift( np.abs(np.fft.ifft2(newEs, [800, 800])) , 0)
        img = np.fft.fftshift( np.abs(np.fft.ifft2(newEs)) , 0)
        if is_vis:
            plot_hrrp(data=img, is_db=False, is_map=True)
        return img

    def _BP_imaging(self, data, is_vis=True):
        # 该BP算法只针对仿真的飞机数据
        ## data 参数
        (hrrp_points, N_phi) = data.shape       # hrrp_points: 256; 孔径数量（有多少个方位角度的观测，10度观测-0.05度角度间隔就是200个点）
        ## 成像系统参数
        fc = 8.5e9
        c = 299792458
        lambda_ = c/fc              # 波长
        k = 2*np.pi / lambda_
        N_prt = 129                 # 步进了多少频率
        delta_f = 1e9/(N_prt - 1)   # 频率步进量
        BW = N_prt * delta_f        # 带宽
        res = c/(2*BW)              # 理论分辨率
        range_unambigous = c/(2*delta_f)        # 相应delta_f，一次处理的不模糊窗距离
        
        ## 成像区域定义
        d_phi = 0.05
        dx = 0.1
        dy = 0.1
        COOR_x = np.linspace(-12, 12, 128)      # 成像坐标
        COOR_y = np.linspace(-12, 12, 128)

        X, Y = np.meshgrid(COOR_x, COOR_y)
        lx = len(COOR_x)
        ly = len(COOR_y)
        # 观测阵列设置
        phi = np.linspace(0, N_phi, N_phi) * d_phi
        Ra = 100
        array_angle = np.column_stack((np.cos(np.radians(phi)), np.sin(np.radians(phi))))
        R_ = np.linspace(-range_unambigous/2, range_unambigous/2, hrrp_points)

        ## 成像
        I = np.zeros([lx, ly]) + 1j*np.zeros([lx, ly])
        for i_dx in range(lx):
            for i_dy in range(ly):
                coor_x = X[i_dx, i_dy]
                coor_y = Y[i_dx, i_dy]
                tmp = 0
                for i_array in range(N_phi):
                    vector = -array_angle[i_array, :]
                    r_ = np.dot([coor_x, coor_y], vector)       # 计算当前像素点和阵列的距离
                    if -range_unambigous/2 <= r_ <= range_unambigous/2:
                        # 如果该距离在模糊距离内部
                        index = np.argmin(np.abs(R_ - r_))  # R_代表一个一维距离像的下标所对应的距离，这句话是找到当前像素点，在当前方位距离像下对应的点下标值
                        cal_ = np.exp(1j * 2 * k * ( range_unambigous / 2 + r_ ))       # 矫正系数
                        tmp += data[index, i_array] * cal_
                # 一轮结束后，将当前像素点像素值放入成像区域中
                I[i_dx, i_dy] = tmp

        ## 绘图
        if is_vis:
            plot_img(I)
        return I
        
    def _OMP_imaging(self, data, is_vis=True):
        """
            解释一下下面的模型：y = Phi * Psi * x
            其中phi是一个随机选择矩阵（这一设定对于不知道孔径的情况下是有效的）
            而已有的数据测量，只是Psi * x，所以需要进行一步线性测量，即Phi * (距离单元全孔径数据)，作为初始值
            参考链接：https://blog.csdn.net/weixin_43475160/article/details/132530899?spm=1001.2014.3001.5502
            参考链接2：https://blog.csdn.net/qq_43095255/article/details/135087322
            参考链接3：https://wenku.csdn.net/answer/6a3yeasfrc
        """
        (hrrp_points, obv_points) = data.shape          # 传入的是距离像
        full_azi_points = 512
        B_hrrp = data
        K = 1                                           # 信号稀疏度
        data_sparse = data.copy()                       # 将传入的数据拷贝一份
        data_rec = np.zeros([hrrp_points, full_azi_points]) + np.zeros([hrrp_points, full_azi_points])*1j
        for coll in range(hrrp_points):
            mmm = data_sparse[coll, :]                  # 取出其中的一行（原因是，现在要恢复的就是方位数据，来替代对方位维度数据做fft）
            # 压缩传感矩阵
            Phi = np.random.randn(obv_points, full_azi_points)      # 测量矩阵  方位维度采样矩阵或者噪声采样
            s = np.dot(Phi, mmm.T)                              # 线性测量  =
            # s = mmm
            # OMP重构信号
            m = 2*K 
            Psi = np.fft.fft(np.eye(full_azi_points, full_azi_points)) / np.sqrt(full_azi_points)   # 512*512
            T = np.dot(Phi, Psi.T)      # 恢复矩阵（测量矩阵*正交反变换矩阵）
            hat_y = np.zeros(full_azi_points) + np.zeros(full_azi_points)*1j
            Aug_t = np.empty((obv_points, 0))
            r_n = s
            for times in range(m):
                product = np.zeros(obv_points)
                for col in range(obv_points):
                    product[col] = np.abs(np.dot(T[:, col].conj().T, r_n))
                pos = np.argmax(product)  # 最大投影系数对应的位置
                Aug_t = np.hstack((Aug_t, T[:, pos][:, np.newaxis]))  # 矩阵扩充
                T[:, pos] = np.zeros(obv_points)  # 选中的列置零（实质上应该去掉，为了简单我把它置零）
                aug_y = np.dot(np.linalg.inv(np.dot(Aug_t.conj().T, Aug_t)), np.dot(Aug_t.conj().T, s))  # 最小二乘,使残差最小
                r_n = s - np.dot(Aug_t, aug_y)  # 残差
            pos_array = np.zeros(m, dtype=int)  # 纪录最大投影系数的位置
            pos_array[:times + 1] = pos
            hat_y[pos_array] = aug_y  # 重构的谱域向量
            data_rec[coll, :] = hat_y

        if is_vis:
            plot_hrrp(data=np.fft.fftshift(data_rec, axes=1), is_db=False, is_map=True)

        return np.fft.fftshift(data_rec, axes=1)

    def _ADMM_imaging(self, data, is_vis=True):
        # TODO
        data = data*1000
        (hrrp_points, obv_points) = data.shape
        full_azi_points = obv_points            # 观测和数据大小一致。目前所有算法均是基于这个假设
        # fft计算矩阵（用来计算对应点数的傅里叶变换）  N x N (N点fft)
        D = np.fft.fft( np.eye(full_azi_points, full_azi_points) )       # 该fft矩阵与被变换矩阵相乘，可以实现对相乘维度的傅里叶变换，点数等于fft矩阵的维度
        measure_matrix = np.random.randn(obv_points, full_azi_points)                               # 测量矩阵，可以是一个random，也可以是一个mask，这个mask的构成参考链接：https://blog.csdn.net/weixin_43475160/article/details/132530899?spm=1001.2014.3001.5502
        # 算子综合
        # A = np.dot(measure_matrix, D)   
        A = D                            # 综合算子
        # 将data做一个转置，迎合fft的计算形式，因为fft矩阵只对0维度做傅里叶变换，所以需要将方位维度放在0维度
        B_hrrp = data.T

        # 初始化成像平面
        x0 = np.random.randn(obv_points, hrrp_points)
        ## 有可能需要使用测量矩阵进行初始化
        # x0 = np.dot(measure_matrix, B_hrrp)
        
        # 将参数送入ADMM算法进行求解
        opts = {}
        opts["verbose"] = 0
        opts["maxit"]   = 100
        opts["sigma"]   = 0.5
        opts["gamma"]   = 10       # gamma 大于1会出现计算超出inf，迭代次数变多后会报错
        opts["ftol"]    = 1e-12
        opts["gtol"]    = 1e-15
        opts["loss"]    = 'l1'
        [rec_img, out] = LASSO_admm_solver(x0, A, B_hrrp, 1e-3, opts)

        if is_vis == True:
            plot_hrrp(data=np.fft.fftshift(rec_img, axes=0).T, is_db=False, is_map=True)
        
        return np.fft.fftshift(rec_img, axes=0).T
        

# 稀疏成像类：获取/生成稀疏数据->使用不同方法成像
class ImagingPlane_Sparse(ImagingPlane):
    def __init__(self, 
                 data_path='data/all_plane_data.npy',
                 ):
        # 调用基类初始化函数
        super(ImagingPlane_Sparse, self).__init__(data_path=data_path)  # 调用父类的初始化函数，将data path传入并读取出数据

    def _hrrp_to_k_space(self, data):
        (hrrp_points, phi_points) = data.shape
        pass
    def _get_sparse_data(self, 
                         plane_num=0, 
                         ele=0, 
                         azi_start=0, 
                         azi_end=10, 
                         azi_step=1, 
                         sparse_method="random choose", 
                         sparse_ratiao=0.5, 
                         mask_path="", 
                         return_type="hrrp_type", # or hrrp_type
                         ):
        # 指定飞机类别，角度，稀疏方法，获得稀疏数据（k-space）
        ## 获取指定类别的数据
        data = self._get_data_from_config(plane_num=plane_num, ele=ele, azi_start=azi_start, azi_end=azi_end, azi_step=azi_step)
        (M, N) = data.shape
        if return_type == "kspace_type":
            ## 将数据转化到k-space
            ret_data = self._hrrp_to_k_space(data=data)
        elif return_type == "hrrp_type":
            ret_data = data
            pass
        ## 稀疏
        if sparse_method == "random choose":
            # 随机选取稀疏角度数据
            # 生成mask
            mask = np.zeros([M, N])
            # 随机选取下标
            mask_idx = np.random.randint(0, N, int(np.floor(N*sparse_ratiao)))
            mask[:, mask_idx] = 1
        elif sparse_method == "mask":
            # TODO
            pass
        else: 
            raise NotImplementedError
        ## 数据乘以mask
        ret_data = ret_data*mask
        return ret_data, mask
    
    def _sparse_imaging_specified_plane(self, 
                                        plane_num=0, 
                                        ele=0, 
                                        azi_start=0, 
                                        azi_end=10, 
                                        azi_step=1, 
                                        sparse_method="random choose", 
                                        sparse_ratiao=0.5, 
                                        mask_path="", 
                                        return_type="kspace_type", # or hrrp_type
                                        method="PD", 
                                        ):
        data, mask = self._get_sparse_data(plane_num=plane_num, ele=ele, 
                                     azi_start=azi_start, azi_end=azi_end, azi_step=azi_step, 
                                     sparse_method=sparse_method, sparse_ratiao=sparse_ratiao, mask_path=mask_path, return_type=return_type)
        self._sparse_imaging(data=data, mask=mask, method=method)
        pass
    # imaging是一个独立的模块，直接接受输入数据，然后进行成像
    def _sparse_imaging(self, data, mask, method="PD", is_vis=True): 
        if method == "PD": 
            img = self._PD_imaging(data, is_vis)
            pass
        elif method == "BP":
            img = self._BP_imaging(data, is_vis)
            pass
        elif method == "PR":
            img = self._PolarRefmt_imaging(data, is_vis)
            pass
        elif method == "OMP":
            img = self._OMP_imaging(data, is_vis)
            pass
        elif method == "ADMM":
            img = self._ADMM_imaging(data, is_vis)
            pass
        elif method == "ADMM-Net":
            # 后续准备练一个ADMM Net！然后就可以直接load保存好的模型，方便他们水论文
            pass
        else:
            raise NotImplementedError

        return img
    

if __name__ == '__main__':
    # 读取和保存hrrp数据测试代码
    # data_root = "D:\\BJC\\DATA\\plane_data"
    # plane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # ele_list = [0]
    # save_path = "data/all_plane_data.npy"
    # plane_hrrp_data = PlaneHRRPData(data_root=data_root, plane_list=plane_list, ele_list=ele_list)
    # # plane_hrrp_data._double_azi()
    # plane_hrrp_data._save_hrrp_data(save_path=save_path)

    # plot_hrrp(plane_hrrp_data.plane_data['0'][:, :, 0], is_db=True, is_map=True)   

    ## imaging
    # data_path = "data/all_plane_data.npy"
    # imging = ImagingPlane(data_path=data_path)
    # imging._imaging_specified_plane(plane_num=7, ele=3, azi_start=3600-128, azi_end=3600+128, azi_step=1, method="BP")

    # test sparse imaging
    data_path = "data/all_plane_data.npy"
    sp_imaging = ImagingPlane_Sparse(data_path=data_path)
    # 稀疏成像
    sp_omp_img = sp_imaging._sparse_imaging_specified_plane(plane_num=7, 
                                               ele=3, 
                                               azi_start=3600-256, 
                                               azi_end=3600+256, 
                                               azi_step=1, 
                                               sparse_method="random choose", 
                                               sparse_ratiao=0.3,
                                               return_type="hrrp_type",
                                               method="ADMM",
                                               ) 
    # 全孔径成像
    pd_img = sp_imaging._imaging_specified_plane(plane_num=7, 
                                        ele=3, 
                                        azi_start=3600-128, 
                                        azi_end=3600+128, 
                                        azi_step=1, 
                                        method="PD",
                                        )
    # 稀疏OMP成像
    sp_pd_img = sp_imaging._sparse_imaging_specified_plane(plane_num=7, 
                                               ele=3, 
                                               azi_start=3600-256, 
                                               azi_end=3600+256, 
                                               azi_step=1, 
                                               sparse_method="random choose", 
                                               sparse_ratiao=0.3,
                                               return_type="hrrp_type",
                                               method="PD",
                                               )
    
    # 
    plot_hrrp(sp_omp_img, is_db=False, is_map=False)
    plot_hrrp(sp_pd_img, is_db=False, is_map=False)
    plot_hrrp(pd_img, is_db=False, is_map=False)

    # github 提交测试
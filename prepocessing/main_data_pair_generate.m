%% 极坐标矫正大角度ISAR成像
clear all
close all
addpath("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging")
% 系统参数设置
c = .3; % speed of light
fc = 8.5; % center frequency
fMin = 8; % lowest frequency
fMax = 9; % highest frequency

phic = 0*pi/180; % center of azimuth look angles
phiMin = -15*pi/180; % lowest angle
phiMax = 15*pi/180; % highest angle

nSampling = 128; % sampling number for integration
nSampling_angle = 600;
% Define Bandwidth
f = fMin:(fMax-fMin)/(nSampling):fMax-(fMax-fMin)/(nSampling);
k = 2*pi*f/.3;
kMax = max(k);
kMin = min(k);
% Define Angle
phi = phiMin:(phiMax-phiMin)/(nSampling_angle):phiMax-(phiMax-phiMin)/(nSampling_angle);

kc = (max(k)+min(k))/2;
 
kx=k.'*cos(phi);
ky=k.'*sin(phi);
 
kxMax = max(max(kx));
kxMin = min(min(kx));
kyMax = max(max(ky));
kyMin = min(min(ky));
 
MM=4; % up sampling ratio
clear kx ky;
kxSteps = (kxMax-kxMin)/(MM*(nSampling+1)-1);
kySteps = (kyMax-kyMin)/(MM*(nSampling_angle+1)-1);
kx = kxMin:kxSteps:kxMax; Nx=length(kx);
ky = kyMin:kySteps:kyMax; Ny=length(ky);
kx(MM*(nSampling+1)+1) = 0;
ky(MM*(nSampling_angle+1)+1) = 0;
% 加载原始数据


% 读取仿真数据HRRP
sel_type = [0 2 3 4 7 9];
% data_path_root = uigetdir('.\');
data_path_root  = "F:\DataSET\plane_data";
azi_start = 345;
azi_rng = 30;
azi_step = 1;
noise_ratio = 0.00;
data_save_path  = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master\data\Reformating_plane";
file_num = 0;
for type_i = 1:length(sel_type)
    type = sel_type(type_i);
    selpath = string(data_path_root) + '\' + num2str(type);

    for ele = 1:10
        file_num = file_num + 1;
        j=1;
        N_phi = 3600;
        N_theta = 1;
        d_phi = 0.05;
        phi_sample_StepPoints = azi_step;
        hrrp_points = 256;
        full_angle = zeros(hrrp_points,N_phi/phi_sample_StepPoints);
        for i = 1:phi_sample_StepPoints:N_phi
            load(selpath + '\TrainingSet\' + num2str(i-1) + '.mat')
            full_angle(:,j) = hrrp(1:1:256,ele);
            j = j+1;
        end
        full_angle_noise =  full_angle + noise_ratio*...
            (randn(hrrp_points,N_phi/phi_sample_StepPoints)  + ...
            1i *randn(hrrp_points,N_phi/phi_sample_StepPoints));            % 添加噪声
        full_angle_noise = [full_angle_noise, fliplr(full_angle_noise)];        % 拼成一张图
        % 挑选其中的三十度进行成像
        N_phi = N_phi/phi_sample_StepPoints;
        d_phi = d_phi*phi_sample_StepPoints;
        shift_size = floor(N_phi * 2 * azi_start / 360);
        full_angle_noise = circshift(full_angle_noise, -shift_size, 2);
        limited_data = full_angle_noise(:, 1:floor(N_phi * 2 * azi_rng / 360)); % 根据平移后的有限角度取对应的数据
        Es = fft(fftshift(limited_data, 1), [], 1);
        Es = Es(1:128, :);
%         Es = Es.';
        %% formatting
        newEs = zeros(MM*(nSampling_angle+1)+1,MM*(nSampling+1)+1);     % 600是角度采样率
        t = 0;
        v = 0;
        for tmpk = k
            t = t+1;
            v = 0;
            for tmpPhi = phi
                v = v+1;
                tmpkx = tmpk*cos(tmpPhi);
                tmpky = tmpk*sin(tmpPhi);
                indexX = floor((tmpkx-kxMin)/kxSteps)+1;
                indexY = floor((tmpky-kyMin)/kySteps)+1;
                
                r1 = sqrt(abs(kx(indexX)-tmpkx)^2+abs(ky(indexY)-tmpky)^2);
                r2 = sqrt(abs(kx(indexX+1)-tmpkx)^2+abs(ky(indexY)-tmpky)^2);
                r3 = sqrt(abs(kx(indexX)-tmpkx)^2+abs(ky(indexY+1)-tmpky)^2);
                r4 = sqrt(abs(kx(indexX+1)-tmpkx)^2+abs(ky(indexY+1)-tmpky)^2);
                
                R = 1/r1+1/r2+1/r3+1/r4;
                
                A1 = Es(t,v)/(r1*R);        
                A2 = Es(t,v)/(r2*R);
                A3 = Es(t,v)/(r3*R);        
                A4 = Es(t,v)/(r4*R);
                     newEs(indexY,indexX) = newEs(indexY,indexX)+A1;
                     newEs(indexY,indexX+1) = newEs(indexY,indexX+1)+A2;
                     newEs(indexY+1,indexX) = newEs(indexY+1,indexX)+A3;
                     newEs(indexY+1,indexX+1) = newEs(indexY+1,indexX+1)+A4;
             end
        end
 
        % down sample newEs by MM times
        newEs=newEs(1:MM: size(newEs, 1),1:MM: size(newEs, 2));
        figure
        imagesc(fftshift(abs(ifft2(newEs))));
        



        %% 根据该数据生成随机稀疏的data.train和data.label
%         data_tmp = newEs.';     % 将矫正后的K域数据做旋转
% %         imagesc(fftshift(fftshift(abs(ifft2(data)), 1), 2))
%         % 生成一个mask
%         mask = zeros(size(data_tmp));
% %         sparse_idx = floor(linspace(1,600,30));
%         sparse_idx = randperm(600, 200);         % 随机排列1:600，然后取前30个数作为下标
%         mask(:, sparse_idx) = 1;
%         data_sparse = data_tmp.*mask;
%         figure
%         imagesc(fftshift(fftshift(abs(ifft2(data_sparse)), 1), 2))            
        %%%% 随机采样的实验证明了使用多的数据和随机采样，可以实现那啥。我们先按照这个思路，验证OMP和ADMM算法的正确性。


        %% 根据这个生成稀疏的data.train 和data.label
        data_tmp = newEs.';     % 将矫正后的K域数据做旋转
%         imagesc(fftshift(fftshift(abs(ifft2(data)), 1), 2))
        % 生成一个mask
        mask = zeros(size(data_tmp));
        sparse_idx = floor(linspace(1,600,30));
        mask(:, sparse_idx) = 1;
        data_sparse = data_tmp.*mask;
        figure
        imagesc(fftshift(fftshift(abs(ifft2(data_tmp)), 1), 2))

        % 均匀采样保留600个点
        % 先在距离维度补零
        pad_mat = zeros(256-size(data_tmp, 1),size(data_tmp, 2));
        data_pad = [data_tmp; pad_mat];
        train = zeros(256, 256);
%         sparse_idx = floor(linspace(1, size(data_tmp, 2), 30));
%         train_idx  = floor(linspace(1, 256, 30));

        sparse_idx = sort(randperm(600, 200));     % 随机采样
%         sparse_idx = floor(linspace(1,600,30));            % 均匀采样
        mask = zeros(size(data_pad));
        mask(:, sparse_idx) = 1;
%         data_sparse = data_pad(:, sparse_idx);
        data_sparse = data_pad;
        data_sparse = data_sparse.*mask;
        train = data_sparse;
        
         %% 均匀采样，顺次拼接后，补零至600点，看一看区别
%         sparse_idx = floor(linspace(1,600,30));
%         train = data_pad(:, sparse_idx);
% %         train = [train, zeros(256, 602-size(train, 2))];
%         
% 
% 
% %         train = [train, zeros(256, 256-size(train, 2))];
%         figure
%         imagesc(abs(train))
% %         train_idx = floor(linspace(1, 256, 200));
% %         train(:, train_idx) = data_pad(:, sparse_idx);          % 取三十列放到这个256列里面
%         % 测试一下这个的成像
%         figure
%         imagesc(fftshift(fftshift(abs(ifft2(train, 256, 256)), 1), 2))
% %         train = [train, zeros(256, 256-size(train, 2))];
        % 存储数据
        data.train = train;

        label_path = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\img\Baseline";
        mat_name = label_path + '\' + num2str(type) + "\小角度相干成像_行归一化\0.003\" + num2str(type) + '_' + num2str(ele * 0.5 + 87) + '_' + num2str(0) + '_' + num2str(0.05) + '_' + num2str(15) + '.mat';
        load(mat_name);
        II = myMaxmin(abs(I + flipud(I)));
        data.label = flipud(fftshift(II.', 2));
        data.label = fftshift(data.label, 1);
        % 建议label还是用原始数据做一个ifft2进行生成吧，但是问题出现在256*256怎么保证
        label_ = ifft2(data_tmp, 256, 602);
%         data.label = imresize(label_, [256,256]);
        data.label = label_;
        data.mask = mask;
        % 保存数据
%         filename = data_save_path + "\" + num2str((ele-1)*6+type_i-1) + ".mat";
%         save(filename, "data");

%         保存到那个最简单的路径吧。
        save_path_2 = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master_256_602\data\Reformating_plane_class_random_sparse" ...
                         + "\" + num2str(type) + "\" + num2str(ele) + ".mat";
        save(save_path_2, "data");
        close all
    end
end









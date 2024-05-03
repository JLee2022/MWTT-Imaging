%% 稀疏成像文件，涵盖了

%% 读取对应的稀疏数据
img_sz = [256,256];
% img_sz = [128,128];
% environment 
addpath('./layersfunction/')
addpath('./layersfunction/line')
addpath('./util')
% addpath('D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master/layersfunction/')
% addpath('D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master/util')
% load mask
if img_sz(1) == 128
    load("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Mask\mask_128.mat");
else
    load("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Mask\mask_256.mat");
end
% data assign
save_root = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\img\CS Methods 256\";
% data_path = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master\data\Reformating_plane\";
% data_path_2 = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\Deep-ADMM-Net-master\data\Reformating_plane\";
data_path_2 = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\data\Reformating_plane_class\";


net_path = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\Train_output\net";
ADMM_NET_data_path = "D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\data\Reformating_plane_class_600_cpx\";
% corr_plane = [0,2,3,4,7,9];     % 对应的飞机类别
corr_plane = [0 2 3 4 7 9 20 26 29];
% corr_plane = [20,26,29];
% corr_plane = [0 2 3 4 7 9];
% net_iter = [580, 630, 500, 1000, 600, 1000];
plane_type = [1,2,3,4,5,6,7:21];
% ele_num = [1,2,3,4,5,6,7,8,9,10];       % 仿真数据
ele_num = [1];                                   % 实验数据
for i = 1:length(corr_plane)
    for j = 1:length(ele_num)
        plane_type_i = i;
        ele_num_i = ele_num(j);
        % load data
%         mat_file = data_path + num2str( (plane_type_i-1)+(ele_num_i-1)*6 ) + ".mat";
        mat_file = data_path_2 + num2str(corr_plane(i)) + "\" + num2str(ele_num_i) + ".mat";
        load(mat_file);     % 数据在data中，有两个标志，一个是label，一个是train-经过遮盖后的回波
        figure
        imagesc((fftshift(myMaxmin(abs(ifft2(data.train))), 2)));
        %% OMP算法成像
        K = 8;      % 信号稀疏度
        [N, M] = size(data.train);
%         N = 602;
%         M = 602;% 256*602时打开
        N = 256; M = 256;
        B_hrrp = ifft(data.train);      % 将竖轴数据做ifft，转化为稀疏的距离像数据
        data_sa = B_hrrp;               % 稀疏数据
        for coll = 1:256
            mmm = data_sa(coll, :);
            % 时域信号压缩传感
            Phi = randn(M, N);          % 测量矩阵
            s = Phi*mmm';               % 线性测量
            % OMP重构信号
            m = 2*K;
            Psi=fft(eye(N,N))/sqrt(N);                        %  傅里叶正变换矩阵
            T=Phi*Psi';                                       %  恢复矩阵(测量矩阵*正交反变换矩阵)
            hat_y=zeros(1,N);                                 %  待重构的谱域(变换域)向量
            Aug_t=[];                                         %  增量矩阵(初始值为空矩阵)
            r_n=s;                                            %  残差值
            for times=1:100                                     %  迭代次数(有噪声的情况下,该迭代次数为K)
                for col=1:N                                   %  恢复矩阵的所有列向量
                    product(col)=abs(T(:,col)'*r_n);          %  恢复矩阵的列向量和残差的投影系数(内积值)
                end
                [val,pos]=max(product);                       %  最大投影系数对应的位置
                Aug_t=[Aug_t,T(:,pos)];                       %  矩阵扩充
                T(:,pos)=zeros(M,1);                          %  选中的列置零（实质上应该去掉，为了简单我把它置零）
                aug_y=(Aug_t'*Aug_t)^(-1)*Aug_t'*s;           %  最小二乘,使残差最小
                r_n=s-Aug_t*aug_y;                            %  残差
                pos_array(times)=pos;                         %  纪录最大投影系数的位置
            end
            hat_y(pos_array)=aug_y;                           %  重构的谱域向量
            data_rec(coll,:)=hat_y;
        end
        im=myfftshit(data_rec);
        imm=fftshift(im);
        g=imm(:,end:-1:1);
        figure;

        %%% 画图
        subplot(131)
        imagesc(myMaxmin(fftshift(abs(data.label), 2)));
        subtitle("Ground Truth");
        subplot(132);
        imagesc(myMaxmin(abs(g)));
        subtitle("OMP");
        subplot(133)
        imagesc((fftshift(myMaxmin(abs(ifft2(data.train))), 2)));
        close all
        % 变量提取，方便保存mat文件
        II = fftshift(abs(g), 1);
        I = myMaxmin(abs(g));
        gt = myMaxmin(fftshift(abs(data.label), 2));
        down_smp = (fftshift(myMaxmin(abs(ifft2(data.train))), 2));
        down_smp_data = ifft(data.train);
        
        fig1 = figure();
        figure(fig1);
        fig1.Position = [300,300,500,500];
        axis normal;

        imagesc(fftshift(I, 1));
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        save_path = save_root + num2str(corr_plane(i)) + "\OMP";
        if ~exist(save_path, 'dir')
            mkdir(save_path)
        end
        fig_name = save_path + "\" + num2str(j) + ".png";
        mat_name = save_path + "\" + num2str(j) + ".mat";
        save(mat_name, "I", "down_smp", "gt", "down_smp_data", "II");
        print(fig_name, '-dpng', '-r600');

        % 输出一个gt ******************************************
        fig1_1 = figure();
        figure(fig1_1);
        fig1_1.Position = [300,300,500,500];
        axis normal;
        imagesc(fftshift(gt, 1));
        colormap jet
        ax=gca;
%         ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        fig_name = save_path + "\" + num2str(j) + "_gt.png";
        print(fig_name, '-dpng', '-r600');

        % 输出一个down sample ******************************************
        fig1_2 = figure();
        figure(fig1_2);
        fig1_2.Position = [300,300,500,500];
        axis normal;
        imagesc(fftshift(down_smp, 1));
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        fig_name = save_path + "\" + num2str(j) + "_ds.png";
        print(fig_name, '-dpng', '-r600');

        % 输出一个sample data ******************************************
        fig1_3 = figure();
        figure(fig1_3);
        fig1_3.Position = [300,300,500,500];
        axis normal;
        imagesc(fftshift(abs(down_smp_data), 1));
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        fig_name = save_path + "\" + num2str(j) + "_dsdata.png";
        print(fig_name, '-dpng', '-r600');
        close all
% 
% 
% 
% 
%         %% ADMM
%         load(mat_file);
%         N = 256;
%         M = 602;
% %         D = fft(eye(602, 602));
%         D = fft(eye(256,256));
% %         mask = data.mask;
% %         mask_trans = mask.';          % 为了迎合矩阵形式的fft方法，需要将mask算子非共轭转置一下
%         % 算子综合
% %         A = mask_trans.*D;            % 将mask和fft算子进行综合
%         
%         A = D;
% %         B_forward = A * data.label.';       % 正向传播，这个传播后就是竖着的稀疏的HRRP数据格式，但是这个数据我们不使用，是传统CT中的方法，和我们的目标数据不是很匹配
% %         B_forward = B_forward .*mask_trans;
%         B_hrrp = ifft(data.train);
%         B_hrrp = B_hrrp.';                  % 为了迎合数据格式将其普通转置一线
%         % ADMM算法参数
%         opts.verbose = 0;
%         opts.maxit = 5000;
%         opts.sigma = 2;
%         opts.gamma = 10;
%         opts.ftol = 1e-12;
%         opts.gtol = 1e-15;
% 
% %         opts.sigma = 1e-2;
% %         opts.ftol = 1e-6;
% %         opts.gtol = 1e-8;
% 
% %         x0 = randn(602, 256);
%         x0 = randn(256,256);
%         % lasso l1 参数
%         lambda = 1;
%         opts.loss = 'l1';
%         tic
%         [rec_img, out] = LASSO_admm_primal(x0,A,B_hrrp,1e-3,opts);
%         toc
%         %%% 画图
%         figure
%         subplot(131) 
%         imagesc(fftshift(myMaxmin(abs(data.label)), 2))
%         title("gt")
%         subplot(132)
%         imagesc(fftshift((myMaxmin(abs(rec_img).')), 2));
%         subtitle("ADMM")
%         subplot(133)
%         imagesc((fftshift(myMaxmin(abs(ifft2(data.train))), 2)));
% 
%         close all
%         fig2 = figure();
%         figure(fig2);
%         fig2.Position = [300,300,500,500];
%         axis normal;
%         % 变量提取，方便保存mat文件
%         II = fftshift(abs(rec_img).',2);
%         I = fftshift((myMaxmin(abs(rec_img).')), 2);
%         gt = myMaxmin(fftshift(abs(data.label), 2));
%         down_smp = (fftshift(myMaxmin(abs(ifft2(data.train))), 2));
%         down_smp_data = ifft(data.train);
% 
%         imagesc(fftshift(I, 1));
%         colormap jet
%         ax=gca;
%         ax.XAxis.Visible='off';
%         ax.YAxis.Visible='off';
%         xlim([-inf inf]);
%         ylim([-inf inf]);
%         xticks([]);yticks([]);
%         set(gca,'LooseInset',get(gca,'TightInset'))
%         save_path = save_root + num2str(corr_plane(i)) + "\ADMM";
%         if ~exist(save_path, 'dir')
%             mkdir(save_path)
%         end
%         fig_name = save_path + "\" + num2str(j) + ".png";
%         mat_name = save_path + "\" + num2str(j) + ".mat";
%         save(mat_name, "I", "down_smp", "gt", "down_smp_data", "II");
%         print(fig_name, '-dpng', '-r600');
% 
%         % 输出一个gt ******************************************
%         fig2_1 = figure();
%         figure(fig2_1);
%         fig2_1.Position = [300,300,500,500];
%         axis normal;
%         imagesc(fftshift(gt, 1));
%         colormap jet
%         ax=gca;
%         ax.XAxis.Visible='off';
%         ax.YAxis.Visible='off';
%         xlim([-inf inf]);
%         ylim([-inf inf]);
%         xticks([]);yticks([]);
%         set(gca,'LooseInset',get(gca,'TightInset'))
%         fig_name = save_path + "\" + num2str(j) + "_gt.png";
%         print(fig_name, '-dpng', '-r150');
% 
%         % 输出一个ds ******************************************
%         fig2_2 = figure();
%         figure(fig2_2);
%         fig2_2.Position = [300,300,500,500];
%         axis normal;
%         imagesc(fftshift(down_smp, 1));
%         colormap jet
%         ax=gca;
%         ax.XAxis.Visible='off';
%         ax.YAxis.Visible='off';
%         xlim([-inf inf]);
%         ylim([-inf inf]);
%         xticks([]);yticks([]);
%         set(gca,'LooseInset',get(gca,'TightInset'))
%         fig_name = save_path + "\" + num2str(j) + "_ds.png";
%         print(fig_name, '-dpng', '-r150');
% 
%         % 输出一个ds data ******************************************
%         fig2_3 = figure();
%         figure(fig2_3);
%         fig2_3.Position = [300,300,500,500];
%         axis normal;
%         imagesc(fftshift(abs(down_smp_data), 1));
%         colormap jet
%         ax=gca;
%         ax.XAxis.Visible='off';
%         ax.YAxis.Visible='off';
%         xlim([-inf inf]);
%         ylim([-inf inf]);
%         xticks([]);yticks([]);
%         set(gca,'LooseInset',get(gca,'TightInset'))
%         fig_name = save_path + "\" + num2str(j) + "_dsdata.png";
%         print(fig_name, '-dpng', '-r150');
%         close all



        %% ADMM-Net
        % load net
%         load(net_path + "net - " + num2str(corr_plane(i)) + "\net-" + num2str(net_iter(i)) + ".mat");
%         load("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\Train_output\六个飞机一起训练（仿真数据集） 256 602\net-00136.mat");
        load("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\Train_output\net\net-00080.mat");
%         load("D:\BaiduNetdiskWorkspace\BigPapers\Code\PreWorking\Coarse2Fine\Imaging\CS-Method\MyCSImaging\ADMM-CSNet-master\Generic-ADMM-CSNet-ComplexMRI\Train_output\net - 副本 7 0俯仰 1e3 5e2\net-00272.mat");
        % load 对应的data
        load(ADMM_NET_data_path + num2str(corr_plane(i)) + "\" + num2str(j) + ".mat");
        y = data.train.*1e3;
%         [re_LOss, rec_image] = loss_with_gradient_single_before(data, net);
        [re_LOss, re_PSnr, rec_image] = before_yloss_with_gradient_single(y, data.label, net);
        % 画图
        % 变量提取
        II = fftshift( fftshift( abs(rec_image), 1 ), 2 );
        I = fftshift(fftshift((abs(rec_image)), 1), 2);
        gt = (fftshift(fftshift(abs(data.label), 2), 1));
        down_smp = fftshift(fftshift((abs(ifft2(data.train))), 2), 1);
        down_smp_data = ifft(data.train);
        
        fig3 = figure();
        figure(fig3);
        fig3.Position = [300,300,500,500];
        axis normal;
        imagesc(I);
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        save_path = save_root + num2str(corr_plane(i)) + "\ADMM-Net";
        if ~exist(save_path, 'dir')
            mkdir(save_path)
        end
        fig_name = save_path + "\" + num2str(j) + ".png";
        mat_name = save_path + "\" + num2str(j) + ".mat";
        save(mat_name, "I", "down_smp", "gt", "down_smp_data", "II");
        print(fig_name, '-dpng', '-r600');

        % 输出一个gt
        fig3_1 = figure();
        figure(fig3_1);
        fig3_1.Position = [300,300,500,500];
        axis normal;
        imagesc(gt);
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        fig_name = save_path + "\" + num2str(j) + "_gt.png";
        print(fig_name, '-dpng', '-r600');

        % 输出一个dsmp img
        fig3_2 = figure();
        figure(fig3_2);
        fig3_2.Position = [300,300,500,500];
        axis normal;
        imagesc(down_smp);
        colormap jet
        ax=gca;
        ax.XAxis.Visible='off';
        ax.YAxis.Visible='off';
        xlim([-inf inf]);
        ylim([-inf inf]);
        xticks([]);yticks([]);
        set(gca,'LooseInset',get(gca,'TightInset'))
        fig_name = save_path + "\" + num2str(j) + "_ds.png";
        print(fig_name, '-dpng', '-r600');
        close all
    end
end

%% ADMM-Net
% addpath('./layersfunction/')
% addpath('./util')
% %% Load trained network
% load('./Train_output/net/net-9076.mat')
% y = data.train;
% %% reconstrction by ADMM-Net
% %tic 
% [re_LOss, rec_image] = loss_with_gradient_single_before(data, net);
% %Time_Net_rec = toc
% re_PSnr = psnr(abs(rec_image) , abs(data.label))
% re_LOss
% Zero_filling_rec = ifft2(y);
% figure;
% subplot(1,3,3); imagesc(abs(Zero_filling_rec)); xlabel('Zero-filling reconstructon result');
% subtitle("GT")
% subplot(1,3,2); imagesc(abs(fftshift(rec_image, 2))); xlabel('ADMM-Net reconstruction result');
% subtitle("ADMM-Net")
% subplot(1,3,1); imagesc(abs(fftshift(data.label, 2))); xlabel('ADMM-Net reconstruction result');
% subtitle("RD")
% % figure
% % imagesc(abs(fftshift(data.label, 2)));
% % figure
% % imagesc(abs(fftshift(rec_image, 2)));
% % imwrite(abs(rec_image),'rec_image.png')
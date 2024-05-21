clear all;
close all;
% start_freq = 28.0;
root =   './1013/'; % root path for raw data 
folder = '34/';
Y = 840;  % folder and param for sub data  % 飞机所在位置中心，即纵坐标
L = 256;  % 截取距离像长度
prefix = 'data_';
postfix = 'GHz.dat';
sweep_freq = 13e9;
N_freq = 251;
sweep_phi = 360;
step_phi = 0.05;

N_phi = sweep_phi/step_phi+1;
step_freq = sweep_freq/(N_freq-1);  
hrrp_points = 1200;
%% step freqency wave param
fc = 33e9;
c = 3e8;
lambda = c/fc;
k = 2*pi/lambda;
N_prt = N_freq;
delta_f = step_freq;
BW = N_prt*delta_f;
res = c/(2*BW);
range_unambigous = c/(2*delta_f);


%%
% FileNamePreprocess(root,folder,prefix,postfix);
%%排序
tmp = dir([root folder]);
tmp ={tmp(3:end).name}.';
B =sort(tmp);

%% preprocess data from far-field software
data_complex = zeros(sweep_phi/step_phi+1,N_freq);
for i=1:N_freq
    file = importdata([root folder char(B(i))]);
    data = file.data;
    tmp_data_mag = 10.^(data(:,2)/10);
    tmp_data_angle = data(:,3)/180*pi;
    data_complex(:,i) = tmp_data_mag.*exp(1i*tmp_data_angle);
end
data_complex = data_complex.'; %[N_freq,N_phi]
%%cal transmit line
data_complex_cal = data_complex .* repmat(exp(1j *2*pi*[0:N_freq-1]*delta_f * (2*1/c)).',1,N_phi);
%%add hamming window
data_complex_cal = data_complex_cal.*repmat(hamming(N_freq),1,N_phi);
%%ifft
full_angle = ifft(data_complex_cal,hrrp_points);
% figure;imagesc(db(full_angle));title([root folder]);
% figure;imagesc(angle(full_angle));title([root folder]);

%%
full_angle_noise = [full_angle];
% full_angle_noise = [full_angle fliplr(full_angle)];

%% plot full angle hrrp
R_ = linspace(0,range_unambigous,hrrp_points);
phi = [0:N_phi-1].'*step_phi;
% figure;mesh(phi,R_,db(full_angle_noise));title([root folder]);
% figure;imagesc(phi,R_,db(full_angle_noise));title([root folder]);
figure;imagesc(db(full_angle_noise));title([root folder]);
% figure;imagesc(angle(full_angle_noise));title([root folder]);
%% jiequ
DataToWrite = full_angle_noise(Y-L/2:Y+L/2-1,:);
figure;imagesc(db(DataToWrite));title([root folder]);

%% 
TestSetPath = ['E:/BJC/Code/Dataset/3_TestSet/experiment' root(2:end) 'hrrp/' folder '0/TrainingSet/'];
if ~isfolder(TestSetPath)
    mkdir(TestSetPath);
end

for i = 1:N_phi-1 % N_phi=7201个点
    hrrp = DataToWrite(:,i);
    save([TestSetPath num2str(i-1) '.mat'],'hrrp'); 
end
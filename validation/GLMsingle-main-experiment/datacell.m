% data1=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run01\lh.surfacedata.mat')
% data2=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run02\lh.surfacedata.mat')
% data3=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run03\lh.surfacedata.mat')
% data4=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run04\lh.surfacedata.mat')
% data5=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run05\lh.surfacedata.mat')
% data6=load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run06\lh.surfacedata.mat')
% cellArray = cell(1, 6);
% cellArray{1} = data1;
% cellArray{2} = data2;
% cellArray{3} = data3;
% cellArray{4} = data4;
% cellArray{5} = data5;
% cellArray{6} = data6;

data1 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run01\lh.surfacedata.mat');
data2 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run02\lh.surfacedata.mat');
data3 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run03\lh.surfacedata.mat');
data4 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run04\lh.surfacedata.mat');
data5 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run05\lh.surfacedata.mat');
data6 = load('E:\5个被试原始数据\CPPDATA\GLMdata\sub01原始数据\session02\preprocess\surfacedata\surface_run06\lh.surfacedata.mat');

data = cell(1, 6);
data{1} = data1.vals;  % 根据具体的变量名进行修改
data{2} = data2.vals;
data{3} = data3.vals;
data{4} = data4.vals;
data{5} = data5.vals;
data{6} = data6.vals;
save('E:\sub01_surfdata\session02\data.mat', 'data');


% data = {data1, data2, data3, data4, data5, data6};


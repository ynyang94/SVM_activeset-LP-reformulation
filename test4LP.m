%preprocessing.
%before getting started, I also processed labels on excel file in order to
%simplify the computing. I change "Iris-setosa" by 1; "Iris-verticolor by
%-1.
%read data
clear all
Data=readtable('iris.csv');


%only pick the first 1000 value for this project.
Data=Data(1:101,:);
Data=table2array(Data);
%Define A
A=[Data(:,3),Data(:,4)];

%Define y (label) 
y=Data(:,5);
y(101,1)=-1;

%segement training and test set. 
indx1=randperm(101,80);
indx1=indx1';

A_train=A(indx1,:);
y_train=y(indx1);
O=[1:1:101]';

indx2=setdiff(O,indx1);

A_test=A(indx2,:);
y_test=y(indx2);

% %%start nodeling.
m=size(A_train,1);
n=size(A_train,2);

%% Make a brief EDA
%training data 
indx_pst=find(y_train==1);
indx_ngtv=find(y_train==-1);
% scatter(A_train(indx_pst,1),A_train(indx_pst,2),'b','filled')
% hold on;
% scatter(A_train(indx_ngtv,1),A_train(indx_ngtv,2),'g','filled')
% xlabel('petal length')
% ylabel('petal width')
% title("EDA for iris training data")
% %test data
indx_pst1=find(y_test==1);
indx_ngtv1=find(y_test==-1);
% scatter(A_test(indx_pst,1),A_test(indx_pst,2),'b','filled')
% hold on;
% scatter(A_test(indx_ngtv1,1),A_test(indx_ngtv1,2),'g','filled')
% xlabel('petal length')
% ylabel('petal width')
% title("EDA for iris test data")

%% SVM primal
%diagnolize y
Y=diag(y_train);

rou=0.9;

%formulate standard linear programming problem.

%set "c"
f=[ones(n,1);rou*ones(m,1); zeros(n,1);0];

% b
b=[-1*ones(m,1);zeros(84,1)];
% A
H=[zeros(m,n),eye(m),-Y*A_train, -Y*ones(m,1);
    zeros(m,n),-1*eye(m,m),zeros(m,n),zeros(m,1);
    -1*eye(n), zeros(n,m), -1*eye(n,n), zeros(n,1);
    -1*eye(n), zeros(n,m), eye(n,n),zeros(n,1)];
[sol1,fval]=linprog(f,H,b);
%Compute gamma
gamma1=sol1(85,1)
%Compute vector v
v1=sol1(83:84,1)
%making plot
%generate point
y_axis=[0:0.1:3]';
x_axis1=-gamma1/v1(1,1)*ones(length(y_axis),1);
plot(x_axis1,y_axis,'r',LineWidth=1.5)
hold on;
%generate hyperplane v'*a+\gamma-1=0
x_axis2=-(gamma1-1)/v1(1,1)*ones(length(y_axis),1);
plot(x_axis2,y_axis,'--r',LineWidth=1)
%generate hyperplane v'*a+\gamma+1=0
x_axis3=-gamma1/v1(1,1)-1/v1(1,1)*ones(length(y_axis),1);
plot(x_axis3,y_axis,'--r',LineWidth=1)
%plot compared with original data
scatter(A_test(indx_pst1,1),A_test(indx_pst1,2),'b','filled')

scatter(A_test(indx_ngtv1,1),A_test(indx_ngtv1,2),'g','filled')

% scatter(A_train(indx_pst,1),A_train(indx_pst,2),'b','filled')
% scatter(A_train(indx_ngtv,1),A_train(indx_ngtv,2),'g','filled')

hold off;

legend('best seperate line','hyperplane-1','hyperplan+1')
xlabel('x1')
ylabel('x2')
title('SVM by solving primal linear programming')

%% SVM dual
f_dual=[-ones(m,1);ones(m,1)];
%set equality constraint
b_eq=0;
H_eq=[zeros(1,m), y_train';];
%inequality constraint.
b_dual=[zeros(m,1);rou*ones(m,1);zeros(m,1);zeros(m,1);];

% H_dual=[zeros(m,m),-1*eye(m,m);
%         zeros(m,m),eye(m,m);
%         -eye(m,m),-A_train*A_train'*Y;
%         -eye(m,m),A_train*A_train'*Y;];
    
%revised version of H
H_dual=[zeros(m,m),-1*eye(m,m);
        zeros(m,m),eye(m,m);
        -eye(m,m),-real(sqrtm(A_train*A_train'))*Y;
        -eye(m,m),real(sqrtm(A_train*A_train'))*Y;];

%solve by linprog solver
[sol2,fval2]=linprog(-f_dual,H_dual,b_dual,H_eq,b_eq);
%compute dual variable zeta
zeta=sol2(1:m);
%compute dual variable lambda
lambda=sol2(m+1:length(sol2))
%from dual, recover v.
v2=A_train'*Y*lambda
%because v2 = [0;0]; the rest is unnecessary to do..
%%
%compute index s.t 0<lambda<rou.
% indx3=find(lambda>0 & lambda<rou);
% %compute gamma.
% sum1=zeros(length(indx3),1);
% for j=1:length(indx3)
%     for i=1:m
%         sum1(j,1)=sum1(j,1)+y(i,:)*lambda(i,:)*A(i,:)*A(indx3(j),:)';
%     end
% end
% gamma2=y(indx3)-sum1;

% %making plot of hyperplane & margin.
% y_axis5=-(v2(1,1)/v2(2,1))*x_axis-gamma2(2)/v2(2,1);
% 
% plot(x_axis,y_axis5,'r',LineWidth=1.5)
% hold on;
% y_axis6=-(v2(1,1)/v2(2,1))*x_axis-gamma2(2)/v2(2,1)-1/v2(2,1);
% plot(x_axis,y_axis6,'--r',LineWidth=1)
% 
% y_axis7=-(v2(1,1)/v2(2,1))*x_axis-gamma2(2)/v2(2,1)+1/v2(2,1);
% plot(x_axis,y_axis7,'--r',LineWidth=1)
% 
% scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
% scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
% hold off;
% legend('best seperate line','hyperplane-1','hyperplan+1')
% xlabel('x1')
% ylabel('x2')
% title('SVM by solving dual linear programming')
%     





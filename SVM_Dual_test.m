% instead of generating data, I directly import original dataset generated
% for HW3(a)
rng('default')
%% Data Generation (Same as Primal SVM)
%generate random number A according to
%standard normal distribution.
m=40;
n=2;
A=randn(m,n);
v=[-0.7626;1];
gamma=0.7;
%set value of y
y=zeros(m,1);
for i=1:m
    if A(i,:)*v+gamma>=0
        y(i)=1;
    else
        y(i)=-1;
    end
end
Y=diag(y);

rou=0.4;

indx_pst=find(y==1);
indx_ngtv=find(y==-1);
x_axis=[-3:0.1:4];

% y_axis=0.7626*x_axis-gamma;
% plot(x_axis,y_axis,'r',LineWidth=1.5)
% hold on;
% scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
% scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
% hold off;
% legend('sample linear classifier')
% xlabel('x1')
% ylabel('x2')
% title('Linear Classifier Demo')


%% formulate dual problem as a quadratic programming problem. 
epsilon=1.e-4;
[m,n]=size(A);
H=Y*A*A'*Y;
H=H+epsilon*eye(m,m);
c=-ones(m,1);
AE=y';
bE=0;
AI=[-eye(m,m);eye(m,m)];
bI=[zeros(m,1);rou*ones(m,1)];

%% solve LP to generate starting point x0.
Ee=diag(sign(bE));
EI=diag(sign(bI));
c_lin=[zeros(m,1);zeros(2*m,1);1;ones(2*m,1)];

AE_LP=[AE,zeros(1,2*m),Ee,zeros(1,2*m);
    AI, eye(2*m,2*m),zeros(2*m,1), EI];
BE_LP=[bE;bI];
AI_LP=[zeros(2*m,m),-eye(80,80),zeros(2*m,1), zeros(2*m,2*m);
    zeros(1,m), zeros(1,2*m), -1, zeros(1,2*m);
    zeros(2*m,m), zeros(2*m,2*m), zeros(2*m,1), -eye(2*m,2*m)];

bI_LP=[zeros(2*m,1);0;zeros(2*m,1)];
sol=linprog(c_lin,AI_LP,bI_LP,AE_LP,BE_LP);
%choose the starting point.
%find an initialization of working set.
lambda0=sol(1:m);
%% Find active set
Activeset=find(abs(AI*lambda0)==0);
rk=0;
%try rk with different values.
% while rk<2
%     wIndx=Activeset(randi([1,size(Activeset,1)],1,1),:);
%     check=[AE;AI(wIndx,:)];
%     rk=rank(check);
% end
% wIndx=wIndx'
wIndx=[];
[lambda,lag1,lag2,wIndx_final,flag,it]=MyQP_ActiveSet(H,c,AI,bI,AE,lambda0,wIndx,100,true);
%% Recompute v and \gamma
v1=A'*Y*lambda;
j=find(lambda>0 & lambda < rou);
sum=zeros(size(j,1),1);
for k=1:size(sum)
    for i=1:m
        sum(k,1)=sum(k,1)+y(i)*lambda(i)*A(i,:)*A(j(k,1),:)';
    end
end
gamma1=y(j)-(sum);
gamma1=gamma1(1)
%% make plot

y_axis1=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1);
plot(x_axis,y_axis1,'r',LineWidth=1.5)
hold on;
%%make hyperplane v'*a+gamma+1=0;
y_axis2=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1)-1/v1(2,1);
plot(x_axis,y_axis2,'--r',LineWidth=1)

%%make hyperplane v'*a+gamma-1=0
y_axis3=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1)+1/v1(2,1);
plot(x_axis,y_axis3,'--r',LineWidth=1)

scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
hold off;
legend('best seperate line','hyperplane-1','hyperplan+1')
xlabel('x1')
ylabel('x2')
title('SVM by solving dual QP')


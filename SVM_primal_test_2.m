

%generate random number A according to
%standard normal distribution.
m=40;
n=2;
A1=normrnd(4,1,m/2,n);
A2=normrnd(-4,1,m/2,n);
A=[A1;A2];

v=[1;1];
gamma=-3;
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

rou=100;

indx_pst=find(y==1);
indx_ngtv=find(y==-1);
% make plot.
x_axis=[-6:0.1:8];
y_axis=-x_axis-gamma;
plot(x_axis,y_axis,'r',LineWidth=1.5)
hold on;
scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
hold off;
legend('sample linear classifier')
xlabel('x1')
ylabel('x2')
title('Linear Classifier Demo')


%% formulate quadratic programming.
H=[eye(n,n),zeros(n,1),zeros(n,m);
    zeros(1,n),zeros(1,1),zeros(1,m);
    zeros(m,n),zeros(m,1),zeros(m,m)];
%because this method require H to be P.S.D, thus we change it a little bit.
epsilon=1.e-15;
%add small perturbation.
%Revised Version of H.
H_distb=[eye(n,n),zeros(n,1),zeros(n,m);
    zeros(1,n),epsilon*eye(1,1),zeros(1,m);
    zeros(m,n),zeros(m,1),epsilon*eye(m,m)];
%linear term
c=[zeros(n,1);zeros(1,1); rou*ones(m,1)];
%Inequality matrix AI
AI=[-Y*A, -Y*ones(m,1),-eye(m,m);
    zeros(m,n), zeros(m,1), -eye(m,m);];
%corresponding bI for inequality constraints.
bI=[-ones(m,1);zeros(m,1)];

EI=diag(sign(bI));
%solve following LP problem to find a feasible starting point x0.
c_lin=[zeros(m+n+1,1); zeros(2*m,1); ones(2*m,1)];
A_lin_eq=[AI, eye(2*m,2*m), EI];
B_lin_eq=bI;
A_lin_iq=[zeros(m+n+1,m+n+1),zeros(m+n+1,2*m), zeros(m+n+1,2*m);
    zeros(2*m,m+n+1), -eye(2*m,2*m), zeros(2*m,2*m);
    zeros(2*m,m+n+1), zeros(2*m,2*m), -eye(2*m,2*m)];

b_lin_iq=zeros(5*m+n+1,1);

sol=linprog(c_lin,A_lin_iq,b_lin_iq,A_lin_eq,B_lin_eq);
%find x0
x0=sol(1:43,:);
%generate active set
%Activeset=find(AI*x0==0);
%rank_AS=rank(AI(Activeset,:))
%generate working set.
%wIndx=Activeset(randi([1,size(Activeset,1)],2,1),:);
wIndx=[];
% apply QP solver for SVM primal problem. no AE is required here.
[opt_sol,lag1,lag2,wIndx_final,flag,it]=MyQP_ActiveSet(H_distb,c,AI,bI,[],x0,wIndx,100,true);


%generate solution for svm
v1=opt_sol(1:n,:);
gamma1=opt_sol(n+1,:)
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
title('SVM by solving primal QP')



%% (a)
%set seed
rng('default')
%generate random number A according to
%standard normal distribution.
m=40;
n=2;
A=randn(m,n);

%set a guess of v and \xi
v=[-0.7626;1];
gamma=0.5;
%set value of y
y=zeros(m,1);
for i=1:m
    if A(i,:)*v+gamma>=0
        y(i)=1;
    else
        y(i)=-1;
    end
end
%plot the margin.
x_axis=[-3:0.1:4];
y_axis=0.7626*x_axis-gamma;
%find index
indx_pst=find(y==1);
indx_ngtv=find(y==-1);
%make plot.
% plot(x_axis,y_axis,'r',LineWidth=1.5)
% hold on;
% scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
% scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
% hold off;
% legend('sample linear classifier')
% xlabel('x1')
% ylabel('x2')
% title('Linear Classifier Demo')
%% (b)
%set rou weighting parameter
Y=diag(y);

rou=0.98;

%formulate standard linear programming problem.
f=[ones(2,1);rou*ones(40,1); zeros(2,1);0];

b=[-1*ones(40,1);zeros(44,1)];

H=[zeros(m,n),eye(m),-Y*A, -Y*ones(m,1);
    zeros(m,n),-1*eye(m,m),zeros(m,n),zeros(m,1);
    -1*eye(n), zeros(n,m), -1*eye(n,n), zeros(n,1);
    -1*eye(n), zeros(n,m), eye(n,n),zeros(n,1)];
%LP solver
[sol1,fval]=linprog(f,H,b);
%find gamma and v1.
gamma1=sol1(45,1);
v1=sol1(43:44,1);

%%make hyperplane v'*a+\gamma=0;
% y_axis1=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1);
% plot(x_axis,y_axis1,'r',LineWidth=1.5)
% hold on;
%%make hyperplane v'*a+gamma+1=0;
% y_axis2=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1)-1/v1(2,1);
% plot(x_axis,y_axis2,'--r',LineWidth=1)
% 
%%make hyperplane v'*a+gamma-1=0
% y_axis3=-(v1(1,1)/v1(2,1))*x_axis-gamma1/v1(2,1)+1/v1(2,1);
% plot(x_axis,y_axis3,'--r',LineWidth=1)
% 
% scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
% scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
% hold off;
% legend('best seperate line','hyperplane-1','hyperplan+1')
% xlabel('x1')
% ylabel('x2')
% title('SVM by solving primal linear programming')

%% (c)
%set c
f_dual=[-ones(m,1);ones(m,1)];
%set equality constraint
b_eq=0;
H_eq=[zeros(1,m), y';];
%inequality constraint.
b_dual=[zeros(m,1);rou*ones(m,1);zeros(m,1);zeros(m,1);];

% H_dual=[zeros(m,m),-1*eye(m,m);
%         zeros(m,m),eye(m,m);
%         -eye(m,m),-A*A'*Y;
%         -eye(m,m),A*A'*Y;];
    
%revised version of H
H_dual=[zeros(m,m),-1*eye(m,m);
        zeros(m,m),eye(m,m);
        -eye(m,m),-1/sqrt(2)*real(sqrtm(A*A'))*Y;
        -eye(m,m),1/sqrt(2)*real(sqrtm(A*A'))*Y;];

%solve by linprog solver
[sol2,fval2]=linprog(-f_dual,H_dual,b_dual,H_eq,b_eq);
%compute dual variable zeta
zeta=sol2(1:m);
%compute dual variable lambda
lambda=sol2(m+1:length(sol2))
%from dual, recover v.
v2=A'*Y*lambda
%compute index s.t 0<lambda<rou.
indx3=find(lambda>0 & lambda<rou);

%compute gamma.
sum1=zeros(length(indx3),1);
for j=1:length(indx3)
    for i=1:m
        sum1(j,1)=sum1(j,1)+y(i,:)*lambda(i,:)*A(i,:)*A(indx3(j),:)';
    end
end
gamma2=y(indx3)-sum1;

%making plot of hyperplane & margin.
%make hyperplane v'*a+\gamma=0;
y_axis5=-(v2(1,1)/v2(2,1))*x_axis-gamma2(3)/v2(2,1);

plot(x_axis,y_axis5,'r',LineWidth=1.5)
hold on;
%make v'*a+\gamma+1=0;
y_axis6=-(v2(1,1)/v2(2,1))*x_axis-gamma2(3)/v2(2,1)-1/v2(2,1);
plot(x_axis,y_axis6,'--r',LineWidth=1)
%make v'*a+\gamma-1=0;
y_axis7=-(v2(1,1)/v2(2,1))*x_axis-gamma2(3)/v2(2,1)+1/v2(2,1);
plot(x_axis,y_axis7,'--r',LineWidth=1)

scatter(A(indx_pst,1),A(indx_pst,2),'b','filled')
scatter(A(indx_ngtv,1),A(indx_ngtv,2),'g','filled')
hold off;
legend('best seperate line','hyperplane-1','hyperplan+1')
xlabel('x1')
ylabel('x2')
title('SVM by solving dual linear programming')
    





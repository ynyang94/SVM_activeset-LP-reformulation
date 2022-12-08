function [x,y,z,wIndx,flag,it]=MyQP_ActiveSet(H,c,AI,bI,AE, x, wIndx,maxit,debug)
%Input;
%H is the positive definite matrix in Quadratic programming (n*n matrix)
%c is the coefficient of linear term (n vector)
% AI, the matrix coefficient corresponding to inequality (mI*n)
% bI, the value.
% AE: Coefficient matrix for equality constraint. If no equality constraint, input AE=[]; 
% x: n vector, initial iterate.
% wIndx: MUST be in the form of 1xk.
% maxit: maximal iteration number
% debug(optional):print out intermediate steps.
%Output
%x is the corresponding solution for quadratic programming.
%y, z are corresponding lagrange multiplier.
%wIndx is the resulted working set.
% flag, the state of iteration, flag=2: iteration>maxit or iteration breaks; 
%flag=1 indicates there is a solution. 
%it:#number of iterations

%define the size of the problem
[mI,n]=size(AI);

%generate index
Indx=1:mI;

%% 'AE is empty' case:
if isempty(AE)
        
    %start iteration.
    it=0;
    while it <= maxit

        it=it+1;
        %check the complement of working set.
        nIndx=setdiff(Indx,wIndx);
        %the column number of working set.
        mW=size(wIndx,2);
        %Formulate linear system.(With the additional inequality constraint).
        % PROBLEM!:Original linear system is not solvable because K is not
        % full rank. We need to add small perturbation to K.(epsilon).
        epsilon=1.e-15;
        K=[H,transpose(AI(wIndx,:));
            AI(wIndx,:), epsilon*eye(mW,mW)];
        rhs=[-(c+H'*x);zeros(mW,1)];
        %% Different ways to get solution of linear system
        % Turns out the above way is the best!
        %K=sparse(K);
        %rhs=sparse(rhs);
        %syz=K\rhs;
        %syz=mldivide(K,rhs)
        
        %%
        %compute solution.
        syz=linsolve(K,rhs);
        %get update s and z.
        s=syz(1:n);
        zW=syz(n+1:n+mW);

        if debug
            z=zeros(mI,1);
            z(wIndx)=zW;
            fprintf(' \n')
            fprintf('Iteration:%d\n',it);
            objective=0.5*(x'*H*x)+c'*x;
            fprintf('Objective function is %f \n:',objective);
            fprintf('working set:\n');
            display(wIndx);
            fprintf('Iterate x:\n')
            display(x);
            fprintf('Step s:\n')
            display(s);
            %fprintf('lagrange multipliers y:\n')
            %display(y);
            fprintf('lagrange multipliers z:\n')
            display(z);
            
        end
        %case 2(a)
        resI=AI*(x+s)-bI;
        if all(resI<=0)
            x=x+s;
            %Optimality Condition Satisfy!
            if all(zW>=0)
                it=it+1;
                flag=0;
                z=zeros(mI,1);
                z(wIndx)=zW;
                %show the solution.
                if debug
                    z=zeros(mI,1);
                    z(wIndx)=zW;
                    fprintf(' \n')
                    fprintf('Iteration:%d\n',it);
                    objective=0.5*(x'*H*x)+c'*x;
                    fprintf('Objective function is %f \n:',objective);
                    fprintf('working set:\n');
                    display(wIndx);
                    fprintf('Iterate x:\n')
                    display(x);
                    fprintf('Step s:\n')
                    display(s);
                    %fprintf('lagrange multipliers y:\n')
                    %display(y);
                    fprintf('lagrange multipliers z:\n')
                    display(z);
                    break
                    
                end
               
            else
                %update index.
                [val,j0]=min(zW);
                j0=wIndx(j0);
                wIndx=setdiff(wIndx,j0);
            end
        %case 2(b)
        else
            
            resI=bI-AI*x;
            
            AIs=AI(nIndx,:)*s;
            
            idx=nIndx(find(AIs > 0));
            %idx could also be empty when the matrix size is large.
            %% We need to check whether it is empty or not.
            if isempty(idx)
                break
            end
            %%
            %I found here idx may exceed the scope of AIs
            %So I add AIs_1 for computing.
            AIs_1=AI*s;
            tmp=resI(idx)./AIs_1(idx);
            %find \alpha to make the update feasible.
            alpha=min(tmp);
            %1e-10
            %Turns out strict equality is better.
            i0= idx(find(abs(alpha-tmp)==0));
            i0=i0(1);
            wIndx=union(wIndx,i0);
            x=x+alpha*s;
            
        end
    end

    %iteration  exceed largest number. 
    if it==maxit
        flag=2;
    else
        flag=1;
    end
    
    if debug
        fprintf(' \n')
        fprintf('Iterations:%d\n',it);
        objective=0.5*(x'*H*x)+dot(c,x);
        fprintf('objective function value is %f \n',objective);
        fprintf('working set:\n');
        display(wIndx);
        fprintf('Iterate x:\n')
        display(x);
        y=0;
        %fprintf('lagrange multiplier y:\n');
        %display(y)
        fprintf('lagrange multiplier z:\n')
        display(z);
        
    end
%% if AE is not empty.    
else
    [mE,n]=size(AE);
    %start iteration.
    it=0;
    while it <= maxit

        it=it+1;
        %check the complement of working set.
        nIndx=setdiff(Indx,wIndx);
        %the column number of working set.
        mW=size(wIndx,2);
        %Formulate linear system.(With the additional inequality constraint).
        %PROBLEM!: Here, with AE, the matrix is still singular after
        %several iterations. We also need to add perturbations to the this
        %system of equations.
        %% H MUST be Positive Definite in order to make system singular!
        K=[H,transpose(AE),transpose(AI(wIndx,:));
           AE,zeros(mE,mE), zeros(mE,mW);
            AI(wIndx,:),zeros(mW,mE), zeros(mW,mW)];
        
        rhs=[-(c+H'*x);zeros(mE,1);zeros(mW,1)];
        %% Different ways to solve linear system.
%         K=sparse([H,transpose(AE),transpose(AI(wIndx,:));
%            AE,zeros(mE,mE), zeros(mE,mW);
%             AI(wIndx,:),zeros(mW,mE), zeros(mW,mW)]);
%         
%         rhs=sparse([-(c+H'*x);zeros(mE,1);zeros(mW,1)]);
        %K=sparse(K);
        %rhs=sparse(rhs);
        %syz=mldivide(K,rhs);
        %syz=K\rhs;
        %%
        syz=linsolve(K,rhs);
        %update s.
        s=syz(1:n);
        %divide lagrange multiplier into y and z.
        y=syz(n+1:n+mE);
        if isnan(y)
            break
            y=1
        end
        
        zW=syz(n+mE+1:n+mE+mW);
        %print out solution.
        if debug
            z=zeros(mI,1);
            z(wIndx)=zW;
            fprintf(' \n')
            fprintf('Iteration:%d\n',it);
            objective=0.5*(x'*H*x)+c'*x;
            fprintf('Objective function is %f \n:',objective);
            fprintf('working set:\n');
            display(wIndx);
            fprintf('Iterate x:\n')
            display(x);
            fprintf('Step s:\n')
            display(s);
            fprintf('lagrange multipliers y:\n')
            display(y);
            fprintf('lagrange multipliers z:\n')
            display(z);
            
        end
        %case 2a.
        epsi=100*eps('double')
        resI=AI*(x+s)-bI;
        if all(resI<=0)
            x=x+s;
            %Optimality condition satisfy!
            %small relaxation rather than 0.
            if all(zW>=-eps)
                it=it+1;
                flag=0;
                z=zeros(mI,1);
                z(wIndx)=zW;
                %print out solution.
                if debug
                    z=zeros(mI,1);
                    z(wIndx)=zW;
                    fprintf(' \n')
                    fprintf('Iteration:%d\n',it);
                    objective=0.5*(x'*H*x)+c'*x;
                    fprintf('Objective function is %f \n:',objective);
                    fprintf('working set:\n');
                    display(wIndx);
                    fprintf('Iterate x:\n')
                    display(x);
                    fprintf('Step s:\n')
                    display(s);
                    fprintf('lagrange multipliers y:\n')
                    display(y);
                    fprintf('lagrange multipliers z:\n')
                    display(z);
                    break
                    flag=0;
                end
                
            else
                %Optimality condition not satisfy.
                %continue update index.
                [val,j0]=min(zW);
                j0=wIndx(j0);
                wIndx=setdiff(wIndx,j0);
            end
        else
            %case 2b.
            resI=bI-AI*x;
            AIs=AI(nIndx,:)*s;
            idx=nIndx(find(AIs > 0));
            %After some iterations, idx may be empty.
            %we need to add stop criterion at this step.
            if isempty(idx)
                break
                
            end
            
            %idx may exceed the scope of AIs, so I just add a new AIs_1;
            AIs_1=AI*s;
            tmp=resI(idx)./AIs_1(idx);
            alpha=min(tmp);
            
            %% 1.e-10 do make better performance.
            i0= idx(find(abs(alpha-tmp)<1.e-10));
            i0=i0(1);
            wIndx=union(wIndx,i0);
            x=x+alpha*s;
        end
    end

    %iteration exceed max iterations. 
    %printout the solution.
    if it==maxit
        flag=2;
    else
        flag=1;
    end
    
    if debug
        fprintf(' \n')
        fprintf('Iterations:%d\n',it);
        objective=0.5*(x'*H*x)+dot(c,x);
        fprintf('objective function value is %f \n',objective);
        fprintf('working set:\n');
        display(wIndx);
        fprintf('Iterate x:\n')
        display(x);
        fprintf('lagrange multiplier y:\n');
        display(y)
        fprintf('lagrange multiplier z:\n')
        display(z);
        
    end

end








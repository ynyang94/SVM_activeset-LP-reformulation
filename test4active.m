H=[1,0;
    0,1];
c=[0;0];
%AE=[0,1];
AI=[1,0;
    0,1;
    -1,-3;];

bI=[2;1;-2];
x0=[2;1];
wIndx=[];
bE=1;
MyQP_ActiveSet(H,c,AI,bI,[],x0,wIndx,100,true)
quadprog(H,c,AI,bI)
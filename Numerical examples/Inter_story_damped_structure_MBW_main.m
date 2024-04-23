clc, clear; close all; %warning off;
% unit: kN, mm, 10^6 kg
p=5; q=2;
dt=0.005*q/p;
Ne=7; 
e_scale=[0.5 0.5 1 0.5 0.5 0.5 0.5];
ag=cell(Ne,1);
t=cell(Ne,1);

earthquake_name=["newhall.txt";"sylmar.txt";"elcentro.txt";...
    "rinaldi.txt";"kobe.txt";"jiji.txt";"erzikan.txt"];
% m/s^2
for i=1:Ne
    raw_data=load(earthquake_name(i));
    ag{i}=e_scale(i)*resample(raw_data(:,1),p,q)*10;
    Nt=length(ag{i});
    t{i}=(0:Nt-1)'*dt;
end
clear earthquake_name raw_data Nt;

% cut off the zero points before and after the sequence
Index=20/dt+2;
ag{1}(Index:end)=[]; t{1}(Index:end)=[];
ag{2}(Index:end)=[]; t{2}(Index:end)=[];
ag{7}(Index:end)=[]; t{7}(Index:end)=[];
Index=15/dt+2;
ag{4}(Index:end)=[]; t{4}(Index:end)=[];
Index=5/dt;
ag{5}(1:Index)=[]; t{5}(1:Index)=[]; t{5}=t{5}-5;
Index=20/dt+2;
ag{5}(Index:end)=[]; t{5}(Index:end)=[];
Index=20/dt;
ag{6}(1:Index)=[]; t{6}(1:Index)=[]; t{6}=t{6}-20;
Index=60/dt+2;
ag{6}(Index:end)=[]; t{6}(Index:end)=[];

Ns=5;
m=2e-3*ones(Ns,1); % 10^6 kg
c=8e-3*ones(Ns,1); % kN/mms
k=4*ones(Ns,1); % kN/mm
M=diag(m); C=diag(c); K=diag(k);
Re=eye(Ns);
for i=1:Ns
    if i>1
        K(i-1,i)=-k(i);
        C(i-1,i)=-c(i);
    end
    if i<Ns
        K(i+1,i)=-k(i+1);
        K(i,i)=K(i,i)+k(i+1);
        C(i+1,i)=-c(i+1);
        C(i,i)=C(i,i)+c(i+1);
        Re(i,i+1)=-1;
    end
end
A0=[zeros(Ns) eye(Ns);-M\K -M\C];
C0=eye(2*Ns);
B1=[zeros(Ns,1+Ns);-ones(Ns,1) -M\Re];
D1=zeros(2*Ns,1+Ns);

%% mm kN
A=0.4; gamma=0.4; beta=0.4; n=2;
alpha=4; % 5
CoeB=[A,gamma,beta,n];
k1=1; k2=0.5; c0=0.05;
a=(k1+k2)/c0; b=k1*k2/c0;
CoeZ=[a,b,k1];

u1=cell(Ne,1);
du1=u1; F1=u1; F1_BW=u1; F1_Zener=u1; F1_damper=u1;

for i=1:Ne
    tend=t{i}(end);
    Earthquake=[t{i} ag{i}];
    sim('Inter_story_damped_structure_MBW_reference');
    u1{i}=x;
    du1{i}=v;
    F1_BW{i}=F_BW;
    F1_Zener{i}=F_Zener;
    F1_damper{i}=F1_BW{i}+F1_Zener{i};
    F1{i}=u1{i};
    F1{i}(:,1)=k(1)*u1{i}(:,1)+c(1)*du1{i}(:,1);
    for j=2:Ns
        F1{i}(:,j)=k(j)*(u1{i}(:,j)-u1{i}(:,j-1))+c(j)*(du1{i}(:,j)-du1{i}(:,j-1));
    end
    F1{i}=F1{i}+F1_damper{i};
end

%%
u2=cell(Ne,1);
du2=u2; F2=u2; F2_damper=u2;
for i=1:Ne
    tend=t{i}(end);
    Earthquake=[t{i} ag{i}];
    sim('Inter_story_damped_structure_MBW_LSTM');
    u2{i}=u;
    du2{i}=v;
    F2_damper{i}=Fd;
    F2{i}=u2{i};
    F2{i}(:,1)=k(1)*u2{i}(:,1)+c(1)*du2{i}(:,1);
    for j=2:Ns
        F2{i}(:,j)=k(j)*(u2{i}(:,j)-u2{i}(:,j-1))+c(j)*(du2{i}(:,j)-du2{i}(:,j-1));
    end
    F2{i}=F2{i}+F2_damper{i};
end

%%
cost_func = 'NRMSE';
J=zeros(7,10);
for i=1:Ne
    for j=1:5
        J(i,j)=goodnessOfFit(F2_damper{i}(:,j),F1_damper{i}(:,j),cost_func);
        J(i,j+5)=goodnessOfFit(u1{i}(:,j),u2{i}(:,j),cost_func);
    end
end
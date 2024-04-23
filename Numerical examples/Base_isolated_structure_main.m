clc, clear; close all; %warning off;
% unit: kN, kpa, m, 10^3kg
% Parameters of the superstrucrure and the base
Base_isolated_structure_infor
Base_isolated_structure_plan
MBW_N=size(MBW_locs,1);
MFD_N=size(MFD_locs,1);

R=zeros(24,3); % R is the matrix of earthquake influence coefficients
for i=1:1:8
    R(3*i-2:3*i,:)=eye(3);
end
% R is the matrix of earthquake influence coefficients.
MK=-M\K;MC=-M\C;m_RC=m\R'*C;m_RK=m\R'*K;

%superstucture subsystem
As=[zeros(24,24) eye(24) zeros(24,6);MK-R*m_RK MC-R*m_RC zeros(24,6);zeros(3,51) eye(3);m_RK m_RC zeros(3,6)];
Bs=[zeros(24,6);zeros(24,3) R/m;zeros(3,6);-eye(3) -inv(m)]; Cs=eye(54); Ds=zeros(54,6);
Reduce=zeros(6,54);Reduce(:,49:54)=eye(6);

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
    temp=e_scale(i)*resample(raw_data(:,1),p,q)/100;
    Nt=length(temp);
    t{i}=(0:Nt-1)'*dt;
    ag{i}=zeros(Nt,3);
    ag{i}(:,1)=temp;
    ag{i}(:,2)=e_scale(i)*resample(raw_data(:,2),p,q)/100;
end
clear earthquake_name raw_data Nt;

% cut off the zero points before and after the sequence
Index=20/dt+2;
ag{1}(Index:end,:)=[]; t{1}(Index:end)=[];
ag{2}(Index:end,:)=[]; t{2}(Index:end)=[];
ag{7}(Index:end,:)=[]; t{7}(Index:end)=[];
Index=15/dt+2;
ag{4}(Index:end,:)=[]; t{4}(Index:end)=[];
Index=5/dt;
ag{5}(1:Index,:)=[]; t{5}(1:Index)=[]; t{5}=t{5}-5;
Index=20/dt+2;
ag{5}(Index:end,:)=[]; t{5}(Index:end)=[];
Index=20/dt;
ag{6}(1:Index,:)=[]; t{6}(1:Index)=[]; t{6}=t{6}-20;
Index=60/dt+2;
ag{6}(Index:end,:)=[]; t{6}(Index:end)=[];

uscale=0.1;
Fscale=80;
% mm kN
A0=0.4; gamma=0.4; beta=0.4; n=2;
alpha0=4; % 
CoeB=[A0,gamma,beta,n];
% m kN
k1=1e3; k2=0.5e3; c0=0.05e3;
a=(k1+k2)/c0; b=k1*k2/c0;
CoeZ=[a,b,k1];

% m kpa kN
A=6000e-6; h=10e-3;
A_h=A/h; % pi()*0.45^2/0.50 % geometric parameters of the VE damper: shear area over shear heihgt
%parameters of FDZe model identified by the experiments. G1,G2,yita (kpa).
G1=4.794e3 ;G2=0.479e3; eta=0.248e3; alpha=0.52;  a_0=(G1+G2)/eta;b_1=A_h*G1; b_0=b_1*G2/eta;
k0=0.5e6;
err=0.001;
lim=length(t{6}); % this is the biggest data number of the earthquake, which is jiji earthquake.
dta=dt^alpha;
GL=zeros(lim-1,1);GL(1)=-alpha;UL=lim-1;T=dt^alpha;
for i=2:1:lim-1
    temp=(1+alpha)/i;GL(i)=GL(i-1)*(1-temp);
    if err>abs(temp)
        UL=i-1;
        break;
    end
end
coe=[-a_0*dta,dta,b_0-a_0*b_1,b_1,UL];

xvab1=cell(7,1);
Fb1=cell(7,1);

for j=1:Ne % E_swit
    Earthquake=[t{j} ag{j}];
    tend=t{j}(end);
    sim('Base_isolated_structure_refence');
    xvab1{j}=xva;
    Fb1{j}=F;
end
%%
Reduce2=zeros(3,54);Reduce2(:,49:51)=eye(3); j=1;
xvab2=cell(7,1);
Fb2=cell(7,1);

for j=1:Ne % E_swit
    Earthquake=[t{j} ag{j}];
    tend=t{j}(end);
    sim('Base_isolated_structure_LSTM');
    xvab2{j}=uv;
    Fb2{j}=Fb;
end

%%
cost_func = 'NRMSE';
J=zeros(7,10);
for i=1:Ne
    for j=1:2
        J(i,j)=goodnessOfFit(xvab1{i}(:,48+j),xvab2{i}(:,48+j),cost_func);
        J(i,j+2)=goodnessOfFit(Fb1{i}(:,j),Fb2{i}(:,j),cost_func);
        J(i,j+4)=goodnessOfFit(xvab1{i}(:,j),xvab2{i}(:,j),cost_func);
        J(i,j+6)=goodnessOfFit(xvab1{i}(:,9+j),xvab2{i}(:,9+j),cost_func);
        J(i,j+8)=goodnessOfFit(xvab1{i}(:,21+j),xvab2{i}(:,21+j),cost_func);
    end
end
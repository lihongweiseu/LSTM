clc, clear; close all;
% Unit mm kN
format long
dt=0.002;

A=0.4; gamma=0.4; beta=0.4; n=2;
alpha=4; % 5
coe=[A,gamma,beta,n];
BoucW=@(du,z)-gamma*abs(du).*abs(z).^(n-1).*z-beta*du.*abs(z).^n+A*du;

k1=1; k2=0.5; c1=0.05;
a=(k1+k2)/c1; b=k1*k2/c1;
Zener=@(du,u,f)b*u+k1*du-a*f;

rng('default'); % To guarantee same results for every running
s = rng;
Niir = 8;
fs=1/dt;

BLWN_amp=linspace(5,33,10)';
BLWN_amp_N=length(BLWN_amp);

BLWN_tend=20; %s
BLWN_fend=6; %Hz
BLWN_Nt=floor(BLWN_tend/dt)+1; % sequence length L
BLWN_t=(0:BLWN_Nt-1)'*dt;
BLWN_u=zeros(BLWN_Nt,BLWN_amp_N); % (L,N)

tin=2; % mutiply a 2-s half-window to make the data start from zero
Ntin=floor(tin/dt)*2;
windows=ones(BLWN_Nt,1);
temp=hann(Ntin);
windows(1:Ntin/2)=temp(1:Ntin/2);
windows(end-Ntin/2:end)=temp(end-Ntin/2:end);
iir = designfilt('lowpassiir','FilterOrder',Niir,'HalfPowerFrequency',BLWN_fend,'SampleRate',fs);

for i=1:BLWN_amp_N
    u=wgn(BLWN_Nt,1,BLWN_amp(i));
    BLWN_u(:,i)=filtfilt(iir,u).*windows;
end

du=zeros(BLWN_Nt,BLWN_amp_N);
z=zeros(BLWN_Nt,BLWN_amp_N);
F_Zener=zeros(BLWN_Nt,BLWN_amp_N);
for i=1:BLWN_Nt-1
    du(i+1,:)=(BLWN_u(i+1,:)-BLWN_u(i,:))/dt*2-du(i,:);
    du_mid=(du(i,:)+du(i+1,:))/2;

    k1=BoucW(du(i,:),z(i,:));
    k2=BoucW(du_mid,z(i,:)+k1*dt/2);
    k3=BoucW(du_mid,z(i,:)+k2*dt/2);
    k4=BoucW(du(i+1,:),z(i,:)+k3*dt);
    z(i+1,:)=z(i,:)+(k1+2*k2+2*k3+k4)*dt/6;

    u_mid=(BLWN_u(i,:)+BLWN_u(i+1,:))/2;
    k1=Zener(du(i,:),BLWN_u(i,:),F_Zener(i,:));
    k2=Zener(du_mid,u_mid,F_Zener(i,:)+k1*dt/2);
    k3=Zener(du_mid,u_mid,F_Zener(i,:)+k2*dt/2);
    k4=Zener(du(i+1,:),BLWN_u(i+1,:),F_Zener(i,:)+k3*dt);
    F_Zener(i+1,:)=F_Zener(i,:)+(k1+2*k2+2*k3+k4)*dt/6;
end
F_BoucW=alpha*z;
BLWN_MBW_y_ref=F_BoucW+F_Zener;

%parameters of FDZe model identified by the experiments. G1,G2,yita (Mpa).
A=6000; h=10;
A_h=A/h; % pi()*0.45^2/0.50 % geometric parameters of the VE damper: shear area over shear heihgt
%parameters of FDZ model identified by the experiments. G1,G2,yita (Gpa).
G1=4.794e-3 ;G2=0.479e-3; eta=0.248e-3; alpha=0.52;  a_0=(G1+G2)/eta;b_1=A_h*G1; b_0=b_1*G2/eta;
k0=0.5e-3;

BLWN_Nf=2^nextpow2(BLWN_Nt);
BLWN_u_fre=fft(BLWN_u,BLWN_Nf);
BLWN_wf=zeros(BLWN_Nf,1);
BLWN_MFD_y_ref_fre=zeros(BLWN_Nf,BLWN_amp_N);
for i=1:1:BLWN_Nf
    if i<BLWN_Nf/2+2
        BLWN_wf(i)=(i-1)*2*pi/BLWN_Nf/dt;
    else
        BLWN_wf(i)=(i-BLWN_Nf-1)*2*pi/BLWN_Nf/dt;
    end
    wf_a=(1i*BLWN_wf(i))^alpha;wf_2=BLWN_wf(i)^2;
    BLWN_MFD_y_ref_fre(i,:)=(b_0+b_1*wf_a)/(a_0+wf_a)*BLWN_u_fre(i,:);
end
temp=ifft(BLWN_MFD_y_ref_fre,BLWN_Nf);
BLWN_MFD_y_ref=real(temp(1:BLWN_Nt,:))+k0*BLWN_u.^3;

BLWN_u=BLWN_u(:,[1 3 5 7 9 10]);
BLWN_MBW_y_ref=BLWN_MBW_y_ref(:,[1 3 5 7 9 10]);
BLWN_MFD_y_ref=BLWN_MFD_y_ref(:,[1 3 5 7 9 10]);
save('BLWN_training_data.mat','BLWN_u','BLWN_MBW_y_ref','BLWN_MFD_y_ref');
clc, clear; close all;
% Unit cm N

% load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'mrdamper.mat'));
load('mrdamper.mat');
Nt=length(F);
dt=Ts; u=V; y_ref=F;
clear Ts V F;
t=(0:Nt-1)'*dt;

MR_MATLAB = importNetworkFromONNX('MR.onnx',InputDataFormats='TBC');
input=dlarray(u,'TBC');
temp=extractdata(predict(MR_MATLAB,input));
y_pre=double(temp);

z = iddata(y_ref, u, dt, 'Name', 'MR damper', ...
    'InputName', 'v', 'OutputName', 'f',...
    'InputUnit', 'cm/s', 'OutputUnit', 'N');  
ze = z(1:2000);    % estimation data
Options = nlarxOptions('SearchMethod', 'lm');
Options.SearchOptions.MaxIterations = 50;
Narx = nlarx(ze, [1 3 1], idSigmoidNetwork, Options); Narx.Name = 'Narx';

y_pre_Narx = sim(Narx,u);

cost_func = 'NRMSE';
fit = goodnessOfFit(y_pre,y_ref,cost_func);
fit_Narx = goodnessOfFit(y_pre_Narx,y_ref,cost_func);

%%
h_all=5;
figure; set(gcf,'Units','centimeter','Position',[5 5 8.7 h_all]);
plot(t,y_ref,'k','linewidth',0.5); hold on; grid on;
plot(t,y_pre_Narx,'b--','linewidth',0.5); 
plot(t,y_pre,'r-.','linewidth',0.5);
plot([10 10],[-100 100],'c','LineWidth',1);
set(gca,'fontsize',7,'fontname','times');
xlim([0,18]); set(gca,'XTick',[0 4 8 10 12 16 18]); % xtickangle(0);
ylim([-100,100]); set(gca,'YTick',-100:50:100);
yticklabels(gca, strrep(yticklabels(gca),'-','−'));
text(5,-86,'\fontname{times}Training','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle');
text(14,-86,'\fontname{times}Testing','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle');
box1=legend({'Exp.',['Narx: ',num2str(fit_Narx,'%.4f')],['LSTM: ',num2str(fit,'%.4f')]},'fontsize',7,'location','northwest','Orientation','horizontal');
box1.ItemTokenSize(1) = box1.ItemTokenSize(1)*0.6;
xlabel('Time (s)','fontsize',7);
ylabel('Force (N)','fontsize',7);

inset_vectior = get(gca, 'TightInset');
inset_x = inset_vectior(1);
inset_y = inset_vectior(2);
inset_w = inset_vectior(3);
inset_h = inset_vectior(4);
% OuterPosition position
outer_vector = get(gca, 'OuterPosition');
xl = outer_vector(1) + inset_x; % Move the origin of Position to the origin of TightInset
yb = outer_vector(2) + inset_y;
w = outer_vector(3) - inset_w - inset_x-0.01/h_all; % Reset the width of Position
h = outer_vector(4) - inset_h - inset_y+0.01/h_all; % Reset the height of Position
% Reset Position
set(gca, 'Position', [xl, yb, w, h]);

Legends_x=xl;
Legends_y=yb+h-box1.Position(4);
box1.Position(1:2)=[Legends_x Legends_y];

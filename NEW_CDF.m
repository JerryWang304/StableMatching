d1 = dlmread('sm_average_times.txt',' ');
d2 = dlmread('wh_average_times.txt',' ');
d1(:,length(d1))=[]
d2(:,length(d2))=[]

lw = 4.0;
ms = 10;
fs = 16;

legendkey = {'Stable Matching','Wait and Hop'};
figure;
title('Compare average completion time');
[h1, stat1] = cdfplot(d1);

set(h1, 'LineStyle','-','color','c','LineWidth',lw);
hold on;

[h2,stat2] = cdfplot(d2);
set(h2,'LineStyle','-','color','r','LineWidth',lw);

hold off;

xlabel('Average Completion Time(s)','FontSize', fs, 'FontName', 'Arial');
ylabel('');
legend(legendkey,'Location','SouthEast');
set(gcf,'position',[100 100 636 400]);
set(gca, 'FontSize', fs, 'FontName', 'Arial','YGrid','on');

set(gcf,'PaperPositionMode','auto');
%print('-r0','-depsc', strcat(filename, '.eps'));

clear;
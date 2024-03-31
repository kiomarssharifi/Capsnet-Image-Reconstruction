clearvars;
resultCell = load('result.mat').result;
result = resultCell{1};
for i = 2:length(resultCell)
    result(i) = resultCell{i};
end
% ssim, mse, pcc, index, char_index, kind_index, type, roi, sub, sess,
% value, orginal, reconstruction
ssim = [result.ssim];
mse = [result.mse];
pcc = [result.pcc];

train = find(cellfun(@(x)any(strcmp(x,'train')),{result.type}));
valid = find(cellfun(@(x)any(strcmp(x,'valid')),{result.type}));

sess1 = find(cellfun(@(x)any(strcmp(x,'ses01')),{result.sess}));
sess2 = find(cellfun(@(x)any(strcmp(x,'ses02')),{result.sess}));
sess1 = intersect(sess1,valid);
sess2 = intersect(sess2,valid);

sub1 = find(cellfun(@(x)any(strcmp(x,'sub01')),{result.sub}));
sub2 = find(cellfun(@(x)any(strcmp(x,'sub02')),{result.sub}));
sub3 = find(cellfun(@(x)any(strcmp(x,'sub03')),{result.sub}));
sub4 = find(cellfun(@(x)any(strcmp(x,'sub04')),{result.sub}));
sub1 = intersect(sub1,valid);
sub2 = intersect(sub2,valid);
sub3 = intersect(sub3,valid);
sub4 = intersect(sub4,valid);

sub1sess = intersect(sub1,[sess1,sess2]);
sub2sess = intersect(sub2,[sess1,sess2]);
sub3sess = intersect(sub3,[sess1,sess2]);
sub4sess = intersect(sub4,[sess1,sess2]);

subsess = intersect([sub1,sub2,sub3,sub4],[sess1,sess2]);

v1 = find(cellfun(@(x)any(strcmp(x,'v1')),{result.roi}));
v1v2 = find(cellfun(@(x)any(strcmp(x,'v1.v2')),{result.roi}));
v1v2v3 = find(cellfun(@(x)any(strcmp(x,'v1.v2.v3')),{result.roi}));
v1 = intersect(v1,valid);
v1v2 = intersect(v1v2,valid);
v1v2v3 = intersect(v1v2v3,valid);

good = find(cellfun(@(x)any(strcmp(x,'good')),{result.value}));
bad = find(cellfun(@(x)any(strcmp(x,'bad')),{result.value}));
good = intersect(good,valid);
bad = intersect(bad,valid);

char1 = find(cellfun(@(x)any(x(1)==1),{result.char_index}));
char2 = find(cellfun(@(x)any(x(1)==2),{result.char_index}));
char3 = find(cellfun(@(x)any(x(1)==3),{result.char_index}));
char4 = find(cellfun(@(x)any(x(1)==4),{result.char_index}));
char5 = find(cellfun(@(x)any(x(1)==5),{result.char_index}));
char6 = find(cellfun(@(x)any(x(1)==6),{result.char_index}));
char7 = find(cellfun(@(x)any(x(1)==7),{result.char_index}));
char8 = find(cellfun(@(x)any(x(1)==8),{result.char_index}));
char1 = intersect(char1,valid);
char2 = intersect(char2,valid);
char3 = intersect(char3,valid);
char4 = intersect(char4,valid);
char5 = intersect(char5,valid);
char6 = intersect(char6,valid);
char7 = intersect(char7,valid);
char8 = intersect(char8,valid);

%% roi effect
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'ROI Effect',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(v1)),mean(ssim(v1v2)),mean(ssim(v1v2v3))];
err = [std(ssim(v1))/sqrt(length(v1))...
      ,std(ssim(v1v2))/sqrt(length(v1v2))...
      ,std(ssim(v1v2v3))/sqrt(length(v1v2v3))];
subplot(1,3,1);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(v1),ssim(v1v2));
P(1,2) = p;
[~,p] = ttest2(ssim(v1v2),ssim(v1v2v3));
P(2,3) = p;
[~,p] = ttest2(ssim(v1),ssim(v1v2v3));
P(1,3) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:3,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:3)
xticklabels({'V1','V1V2','V1V2V3'});
title('Avg SSIM Over All Sub and Sessb','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(v1)),mean(mse(v1v2)),mean(mse(v1v2v3))]);
err = [std(mse(v1))/sqrt(length(v1))...
      ,std(mse(v1v2))/sqrt(length(v1v2))...
      ,std(mse(v1v2v3))/sqrt(length(v1v2v3))];
subplot(1,3,2);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(v1),mse(v1v2));
P(1,2) = p;
[~,p] = ttest2(mse(v1v2),mse(v1v2v3));
P(2,3) = p;
[~,p] = ttest2(mse(v1),mse(v1v2v3));
P(1,3) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:3,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:3)
xticklabels({'V1','V1V2','V1V2V3'});
ylabel('MSE');
title('Avg MSE Over All Sub and Sess','FontSize',13);

y = [mean(pcc(v1)),mean(pcc(v1v2)),mean(pcc(v1v2v3))];
err = [std(pcc(v1))/sqrt(length(v1))...
      ,std(pcc(v1v2))/sqrt(length(v1v2))...
      ,std(pcc(v1v2v3))/sqrt(length(v1v2v3))];
subplot(1,3,3);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(v1),pcc(v1v2));
P(1,2) = p;
[~,p] = ttest2(pcc(v1v2),pcc(v1v2v3));
P(2,3) = p;
[~,p] = ttest2(pcc(v1),pcc(v1v2v3));
P(1,3) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:3,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:3)
xticklabels({'V1','V1V2','V1V2V3'});
ylabel('PCC');
title('Avg PCC Over All Sub and Sess','FontSize',13);
saveas(f,'result/roi.eps');

%% sub diffrence
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'Subject Diffrence',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(sub1sess)),mean(ssim(sub2sess))...
    ,mean(ssim(sub3sess)),mean(ssim(sub4sess))];
err = [std(ssim(sub1sess))/sqrt(length(sub1sess))...
      ,std(ssim(sub2sess))/sqrt(length(sub2sess))...
      ,std(ssim(sub3sess))/sqrt(length(sub3sess))...
      ,std(ssim(sub4sess))/sqrt(length(sub4sess))];
subplot(1,3,1);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(sub1sess),ssim(sub2sess));
P(1,2) = p;
[~,p] = ttest2(ssim(sub1sess),ssim(sub3sess));
P(1,3) = p;
[~,p] = ttest2(ssim(sub1sess),ssim(sub4sess));
P(1,4) = p;
[~,p] = ttest2(ssim(sub2sess),ssim(sub3sess));
P(2,3) = p;
[~,p] = ttest2(ssim(sub2sess),ssim(sub4sess));
P(2,4) = p;
[~,p] = ttest2(ssim(sub3sess),ssim(sub4sess));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:4,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:4);
xticklabels({'sub1','sub2','sub3','sub4'});
ylabel('SSIM');
title('Avg SSIM Over All Sess','FontSize',13);
hold off

y = double([mean(mse(sub1sess)),mean(mse(sub2sess))...
           ,mean(mse(sub3sess)),mean(mse(sub4sess))]);
err = [std(mse(sub1sess))/sqrt(length(sub1sess))...
      ,std(mse(sub2sess))/sqrt(length(sub2sess))...
      ,std(mse(sub3sess))/sqrt(length(sub3sess))...
      ,std(mse(sub4sess))/sqrt(length(sub4sess))];
subplot(1,3,2);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(sub1sess),mse(sub2sess));
P(1,2) = p;
[~,p] = ttest2(mse(sub1sess),mse(sub3sess));
P(1,3) = p;
[~,p] = ttest2(mse(sub1sess),mse(sub4sess));
P(1,4) = p;
[~,p] = ttest2(mse(sub2sess),mse(sub3sess));
P(2,3) = p;
[~,p] = ttest2(mse(sub2sess),mse(sub4sess));
P(2,4) = p;
[~,p] = ttest2(mse(sub3sess),mse(sub4sess));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:4,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:4);
xticklabels({'sub1','sub2','sub3','sub4'});
ylabel('MSE');
title('Avg MSE Over All Sess','FontSize',13);
hold off

y = [mean(pcc(sub1sess)),mean(pcc(sub2sess))...
    ,mean(pcc(sub3sess)),mean(pcc(sub4sess))];
err = [std(pcc(sub1sess))/sqrt(length(sub1sess))...
      ,std(pcc(sub2sess))/sqrt(length(sub2sess))...
      ,std(pcc(sub3sess))/sqrt(length(sub3sess))...
      ,std(pcc(sub4sess))/sqrt(length(sub4sess))];
subplot(1,3,3);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(sub1sess),pcc(sub2sess));
P(1,2) = p;
[~,p] = ttest2(pcc(sub1sess),pcc(sub3sess));
P(1,3) = p;
[~,p] = ttest2(pcc(sub1sess),pcc(sub4sess));
P(1,4) = p;
[~,p] = ttest2(pcc(sub2sess),pcc(sub3sess));
P(2,3) = p;
[~,p] = ttest2(pcc(sub2sess),pcc(sub4sess));
P(2,4) = p;
[~,p] = ttest2(pcc(sub3sess),pcc(sub4sess));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:4,y,'E',err,'P',P,'ErrorbarStyle','T');
xticks(1:4);
xticklabels({'sub1','sub2','sub3','sub4'});
ylabel('PCC');
title('Avg PCC Over All Sess','FontSize',13);
saveas(f,'result/sub.eps');

%% char diffrence
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'Char Diffrence',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(char1)),mean(ssim(char2)),mean(ssim(char3))...
    ,mean(ssim(char4)),mean(ssim(char5)),mean(ssim(char6))...
    ,mean(ssim(char7)),mean(ssim(char8))];
err = [std(ssim(char1))/sqrt(length(char1))...
      ,std(ssim(char2))/sqrt(length(char2))...
      ,std(ssim(char3))/sqrt(length(char3))...
      ,std(ssim(char4))/sqrt(length(char4))...
      ,std(ssim(char5))/sqrt(length(char5))...
      ,std(ssim(char6))/sqrt(length(char6))...
      ,std(ssim(char7))/sqrt(length(char7))...
      ,std(ssim(char8))/sqrt(length(char8))];
subplot(1,3,1);
b = superbar(1:8,y,'E',err,'ErrorbarStyle','T');
xticks(1:8);
xticklabels({'char1','char2','char3','char4'...
            ,'char5','char6','char7','char8'});
ylabel('SSIM');
title('Avg SSIM Over All Sess and Sess','FontSize',13);

y = [mean(mse(char1)),mean(mse(char2)),mean(mse(char3))...
    ,mean(mse(char4)),mean(mse(char5)),mean(mse(char6))...
    ,mean(mse(char7)),mean(mse(char8))];
err = [std(mse(char1))/sqrt(length(char1))...
      ,std(mse(char2))/sqrt(length(char2))...
      ,std(mse(char3))/sqrt(length(char3))...
      ,std(mse(char4))/sqrt(length(char4))...
      ,std(mse(char5))/sqrt(length(char5))...
      ,std(mse(char6))/sqrt(length(char6))...
      ,std(mse(char7))/sqrt(length(char7))...
      ,std(mse(char8))/sqrt(length(char8))];
subplot(1,3,2);
b = superbar(1:8,y,'E',err,'ErrorbarStyle','T');
xticks(1:8);
xticklabels({'char1','char2','char3','char4'...
            ,'char5','char6','char7','char8'});
ylabel('MSE');
title('Avg MSE Over All Sess and Sess','FontSize',13);

y = [mean(pcc(char1)),mean(pcc(char2)),mean(pcc(char3))...
    ,mean(pcc(char4)),mean(pcc(char5)),mean(pcc(char6))...
    ,mean(pcc(char7)),mean(pcc(char8))];
err = [std(pcc(char1))/sqrt(length(char1))...
      ,std(pcc(char2))/sqrt(length(char2))...
      ,std(pcc(char3))/sqrt(length(char3))...
      ,std(pcc(char4))/sqrt(length(char4))...
      ,std(pcc(char5))/sqrt(length(char5))...
      ,std(pcc(char6))/sqrt(length(char6))...
      ,std(pcc(char7))/sqrt(length(char7))...
      ,std(pcc(char8))/sqrt(length(char8))];
subplot(1,3,3);
b = superbar(1:8,y,'E',err,'ErrorbarStyle','T');
xticks(1:8);
xticklabels({'char1','char2','char3','char4'...
            ,'char5','char6','char7','char8'});
ylabel('PCC');
title('Avg PCC Over All Sess and Sess','FontSize',13);
saveas(f,'result/char.eps');

%% value vs Sess
f = figure('units','normalized','outerposition',[0 0 1 1]);

good_sess1 = intersect(good,sess1);
bad_sess1 = intersect(bad,sess1);

good_sess2 = intersect(good,sess2);
bad_sess2 = intersect(bad,sess2);

annotation('textbox',[.35 .9 .1 .1],'String',...
    'Effect of Value Traning on Reconstruction Quality',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(bad_sess1)),mean(ssim(bad_sess2));...
     mean(ssim(good_sess1)),mean(ssim(good_sess2))];
err = [std(ssim(good_sess1))/sqrt(length(good_sess1))...
      ,std(ssim(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(ssim(good_sess2))/sqrt(length(good_sess2))...
      ,std(ssim(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,1);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(bad_sess1),ssim(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(ssim(bad_sess1),ssim(good_sess1));
P(1,3) = p;
[~,p] = ttest2(ssim(bad_sess2),ssim(good_sess2));
P(2,4) = p;
[~,p] = ttest2(ssim(good_sess1),ssim(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg SSIM Over All Sub','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(bad_sess1)),mean(mse(bad_sess2));...
     mean(mse(good_sess1)),mean(mse(good_sess2))]);
err = [std(mse(good_sess1))/sqrt(length(good_sess1))...
      ,std(mse(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(mse(good_sess2))/sqrt(length(good_sess2))...
      ,std(mse(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,2);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(bad_sess1),mse(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(mse(bad_sess1),mse(good_sess1));
P(1,3) = p;
[~,p] = ttest2(mse(bad_sess2),mse(good_sess2));
P(2,4) = p;
[~,p] = ttest2(mse(good_sess1),mse(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg MSE Over All Sub','FontSize',13);
ylabel('MSE');

y = [mean(pcc(bad_sess1)),mean(pcc(bad_sess2));...
     mean(pcc(good_sess1)),mean(pcc(good_sess2))];
err = [std(pcc(good_sess1))/sqrt(length(good_sess1))...
      ,std(pcc(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(pcc(good_sess2))/sqrt(length(good_sess2))...
      ,std(pcc(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,3);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(bad_sess1),pcc(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(pcc(bad_sess1),pcc(good_sess1));
P(1,3) = p;
[~,p] = ttest2(pcc(bad_sess2),pcc(good_sess2));
P(2,4) = p;
[~,p] = ttest2(pcc(good_sess1),pcc(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg PCC Over All Sub','FontSize',13);
ylabel('PCC');
saveas(f,'result/value.eps','epsc');
%% value vs Sess
f = figure('units','normalized','outerposition',[0 0 1 1]);
roi_name = 'v1';
roi_var = v1;
temp = intersect(roi_var,sess1);
good_sess1 = intersect(good,temp);
bad_sess1 = intersect(bad,temp);

temp = intersect(roi_var,sess1);
good_sess2 = intersect(good,temp);
bad_sess2 = intersect(bad,temp);

annotation('textbox',[.35 .9 .1 .1],'String',...
    ['Effect of Value Traning on Reconstruction Quality (',num2str(roi_name),')'],...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(bad_sess1)),mean(ssim(bad_sess2));...
     mean(ssim(good_sess1)),mean(ssim(good_sess2))];
err = [std(ssim(good_sess1))/sqrt(length(good_sess1))...
      ,std(ssim(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(ssim(good_sess2))/sqrt(length(good_sess2))...
      ,std(ssim(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,1);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(bad_sess1),ssim(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(ssim(bad_sess1),ssim(good_sess1));
P(1,3) = p;
[~,p] = ttest2(ssim(bad_sess2),ssim(good_sess2));
P(2,4) = p;
[~,p] = ttest2(ssim(good_sess1),ssim(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg SSIM Over All Sub','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(bad_sess1)),mean(mse(bad_sess2));...
     mean(mse(good_sess1)),mean(mse(good_sess2))]);
err = [std(mse(good_sess1))/sqrt(length(good_sess1))...
      ,std(mse(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(mse(good_sess2))/sqrt(length(good_sess2))...
      ,std(mse(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,2);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(bad_sess1),mse(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(mse(bad_sess1),mse(good_sess1));
P(1,3) = p;
[~,p] = ttest2(mse(bad_sess2),mse(good_sess2));
P(2,4) = p;
[~,p] = ttest2(mse(good_sess1),mse(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg MSE Over All Sub','FontSize',13);
ylabel('MSE');

y = [mean(pcc(bad_sess1)),mean(pcc(bad_sess2));...
     mean(pcc(good_sess1)),mean(pcc(good_sess2))];
err = [std(pcc(good_sess1))/sqrt(length(good_sess1))...
      ,std(pcc(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(pcc(good_sess2))/sqrt(length(good_sess2))...
      ,std(pcc(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,3);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(bad_sess1),pcc(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(pcc(bad_sess1),pcc(good_sess1));
P(1,3) = p;
[~,p] = ttest2(pcc(bad_sess2),pcc(good_sess2));
P(2,4) = p;
[~,p] = ttest2(pcc(good_sess1),pcc(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg PCC Over All Sub','FontSize',13);
ylabel('PCC');
saveas(f,['result/value_',num2str(roi_name),'.eps'],'epsc');
%% value vs Sess Per Sub
f = figure('units','normalized','outerposition',[0 0 1 1]);
sub_index = 1;
sub = sub1;

temp = intersect(sub,sess1);
good_sess1 = intersect(good,temp);
bad_sess1 = intersect(bad,temp);

temp = intersect(sub,sess2);
good_sess2 = intersect(good,temp);
bad_sess2 = intersect(bad,temp);

annotation('textbox',[.35 .9 .1 .1],'String',...
    ['Effect of Value Traning on Reconstruction Quality (Sub',num2str(sub_index),')'],...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(bad_sess1)),mean(ssim(bad_sess2));...
     mean(ssim(good_sess1)),mean(ssim(good_sess2))];
err = [std(ssim(good_sess1))/sqrt(length(good_sess1))...
      ,std(ssim(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(ssim(good_sess2))/sqrt(length(good_sess2))...
      ,std(ssim(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,1);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(bad_sess1),ssim(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(ssim(bad_sess1),ssim(good_sess1));
P(1,3) = p;
[~,p] = ttest2(ssim(bad_sess2),ssim(good_sess2));
P(2,4) = p;
[~,p] = ttest2(ssim(good_sess1),ssim(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg SSIM Over single Sub','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(bad_sess1)),mean(mse(bad_sess2));...
     mean(mse(good_sess1)),mean(mse(good_sess2))]);
err = [std(mse(good_sess1))/sqrt(length(good_sess1))...
      ,std(mse(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(mse(good_sess2))/sqrt(length(good_sess2))...
      ,std(mse(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,2);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(bad_sess1),mse(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(mse(bad_sess1),mse(good_sess1));
P(1,3) = p;
[~,p] = ttest2(mse(bad_sess2),mse(good_sess2));
P(2,4) = p;
[~,p] = ttest2(mse(good_sess1),mse(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg MSE Over single Sub','FontSize',13);
ylabel('MSE');

y = [mean(pcc(bad_sess1)),mean(pcc(bad_sess2));...
     mean(pcc(good_sess1)),mean(pcc(good_sess2))];
err = [std(pcc(good_sess1))/sqrt(length(good_sess1))...
      ,std(pcc(bad_sess1)) /sqrt(length(bad_sess1)) ...
      ;std(pcc(good_sess2))/sqrt(length(good_sess2))...
      ,std(pcc(bad_sess2)) /sqrt(length(bad_sess2))];
  
subplot(1,3,3);
C = [0.6350 0.0780 0.1840
     0.6350 0.0780 0.1840
     0.4660 0.6740 0.1880
     0.4660 0.6740 0.1880];
C = reshape(C, [2 2 3]);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(bad_sess1),pcc(bad_sess2));
P(1,2) = p;
[~,p] = ttest2(pcc(bad_sess1),pcc(good_sess1));
P(1,3) = p;
[~,p] = ttest2(pcc(bad_sess2),pcc(good_sess2));
P(2,4) = p;
[~,p] = ttest2(pcc(good_sess1),pcc(good_sess2));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar([1,2],y,'E',err,'P',P,'BarFaceColor',C,'ErrorbarStyle','T');
xticks([1,2])
xticklabels({'Pre','Post'})
legend([b(1),b(3)],{'Bad','Good'});
legend('boxoff');
title('Avg PCC Over single Sub','FontSize',13);
ylabel('PCC');
saveas(f,['result/value_sub',num2str(sub_index),'.eps'],'epsc');
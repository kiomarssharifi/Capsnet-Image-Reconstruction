clearvars;
clc;
resultCell1 = load('result1.mat').result;
resultCell2 = load('result2.mat').result;
resultCell3 = load('result3.mat').result;
resultCell4 = load('result4.mat').result;
resultCell5 = load('result5.mat').result;
resultCell6 = load('result6.mat').result;
resultCell = cat(2,resultCell1,resultCell2,resultCell3,resultCell4,resultCell5,resultCell6);
result = resultCell{1};
for i = 2:length(resultCell)
    result(i) = resultCell{i};
end
% ssim, mse, pcc, index, char_index, kind_index, type, roi, sub, sess,
% value, orginal, reconstruction
ssim = nan(1,length(result));
for i=1:length(result)
   ssim(i) = result(i).ssim;
end
mse = nan(1,length(result));
for i=1:length(result)
   mse(i) = result(i).mse;
end
pcc = nan(1,length(result));
for i=1:length(result)
   pcc(i) = result(i).pcc;
end
lr = nan(1,length(result));
for i=1:length(result)
   lr(i) = result(i).lr;
end
lrUnique = unique(lr);
wd = nan(1,length(result));
for i=1:length(result)
   wd(i) = result(i).wd;
end
wdUnique = unique(wd);
bs = nan(1,length(result));
for i=1:length(result)
   bs(i) = result(i).bs;
end
bsUnique = unique(bs);

train = find(cellfun(@(x)any(strcmp(x,'train')),{result.type}));
valid = find(cellfun(@(x)any(strcmp(x,'valid')),{result.type}));

lr1 = find(lr == lrUnique(1));
lr2 = find(lr == lrUnique(2));
lr3 = find(lr == lrUnique(3));
lr4 = find(lr == lrUnique(4));
lr5 = find(lr == lrUnique(5));
lr6 = find(lr == lrUnique(6));
lr1 = intersect(lr1,valid);
lr2 = intersect(lr2,valid);
lr3 = intersect(lr3,valid);
lr4 = intersect(lr4,valid);
lr5 = intersect(lr5,valid);
lr6 = intersect(lr6,valid);

wd1 = find(wd == wdUnique(1));
wd2 = find(wd == wdUnique(2));
wd3 = find(wd == wdUnique(3));
wd4 = find(wd == wdUnique(4));
wd5 = find(wd == wdUnique(5));
wd6 = find(wd == wdUnique(6));
wd7 = find(wd == wdUnique(7));
wd8 = find(wd == wdUnique(8));
wd1 = intersect(wd1,valid);
wd2 = intersect(wd2,valid);
wd3 = intersect(wd3,valid);
wd4 = intersect(wd4,valid);
wd5 = intersect(wd5,valid);
wd6 = intersect(wd6,valid);
wd7 = intersect(wd7,valid);
wd8 = intersect(wd8,valid);

bs1 = find(bs == bsUnique(1));
bs2 = find(bs == bsUnique(2));
bs3 = find(bs == bsUnique(3));
bs4 = find(bs == bsUnique(4));
bs5 = find(bs == bsUnique(5));
bs1 = intersect(bs1,valid);
bs2 = intersect(bs2,valid);
bs3 = intersect(bs3,valid);
bs4 = intersect(bs4,valid);
bs5 = intersect(bs5,valid);
%%
lr_wd_ssim = nan(length(lrUnique),length(wdUnique),length(bsUnique));
lr_wd_mse = nan(length(lrUnique),length(wdUnique),length(bsUnique));
lr_wd_pcc = nan(length(lrUnique),length(wdUnique),length(bsUnique));
for ilr = 1:length(lrUnique)
    for iwd = 1:length(wdUnique)
        for ibs = 1:length(bsUnique)
            lrtemp = lrUnique(ilr);
            wdtemp = wdUnique(iwd);
            bstemp = bsUnique(ibs);
            ix = find(lr == lrtemp & wd == wdtemp & bs == bstemp);
            ix = intersect(ix,valid);
            lr_wd_ssim(ilr,iwd,ibs) = mean(ssim(ix));
            lr_wd_mse(ilr,iwd,ibs) = mean(mse(ix));
            lr_wd_pcc(ilr,iwd,ibs) = mean(pcc(ix));
        end
    end
end
[M,I] = max(lr_wd_ssim(:))
[i,j,k] = ind2sub(size(lr_wd_ssim),I)
lrUnique(i)
wdUnique(j)
bsUnique(k)

%% lr effect
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'LR Effect',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(lr1)),mean(ssim(lr2)),mean(ssim(lr3))...
    ,mean(ssim(lr4)),mean(ssim(lr5)),mean(ssim(lr6))];
err = [std(ssim(lr1))/sqrt(length(lr1))...
      ,std(ssim(lr2))/sqrt(length(lr2))...
      ,std(ssim(lr3))/sqrt(length(lr3))...
      ,std(ssim(lr4))/sqrt(length(lr4))...
      ,std(ssim(lr5))/sqrt(length(lr5))...
      ,std(ssim(lr6))/sqrt(length(lr6))];
subplot(1,3,1);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(lr1),ssim(lr2));
P(1,2) = p;
[~,p] = ttest2(ssim(lr1),ssim(lr3));
P(1,3) = p;
[~,p] = ttest2(ssim(lr1),ssim(lr4));
P(1,4) = p;
[~,p] = ttest2(ssim(lr2),ssim(lr3));
P(2,3) = p;
[~,p] = ttest2(ssim(lr2),ssim(lr4));
P(2,4) = p;
[~,p] = ttest2(ssim(lr3),ssim(lr4));
P(3,4) = p;

% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(lrUnique(1)),num2str(lrUnique(2))...
            ,num2str(lrUnique(3)),num2str(lrUnique(4))...
            ,num2str(lrUnique(5)),num2str(lrUnique(6))});
title('Avg SSIM Over All Sub and Sessb','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(lr1)),mean(mse(lr2)),mean(mse(lr3))...
    ,mean(mse(lr4)),mean(mse(lr5)),mean(mse(lr6))]);
err = [std(mse(lr1))/sqrt(length(lr1))...
      ,std(mse(lr2))/sqrt(length(lr2))...
      ,std(mse(lr3))/sqrt(length(lr3))...
      ,std(mse(lr4))/sqrt(length(lr4))...
      ,std(mse(lr5))/sqrt(length(lr5))...
      ,std(mse(lr6))/sqrt(length(lr6))];
subplot(1,3,2);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(lr1),mse(lr2));
P(1,2) = p;
[~,p] = ttest2(mse(lr1),mse(lr3));
P(1,3) = p;
[~,p] = ttest2(mse(lr1),mse(lr4));
P(1,4) = p;
[~,p] = ttest2(mse(lr2),mse(lr3));
P(2,3) = p;
[~,p] = ttest2(mse(lr2),mse(lr4));
P(2,4) = p;
[~,p] = ttest2(mse(lr3),mse(lr4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(lrUnique(1)),num2str(lrUnique(2))...
            ,num2str(lrUnique(3)),num2str(lrUnique(4))...
            ,num2str(lrUnique(5)),num2str(lrUnique(6))});
ylabel('MSE');
title('Avg MSE Over All Sub and Sess','FontSize',13);

y = [mean(pcc(lr1)),mean(pcc(lr2)),mean(pcc(lr3))...
    ,mean(pcc(lr4)),mean(pcc(lr5)),mean(pcc(lr6))];
err = [std(pcc(lr1))/sqrt(length(lr1))...
      ,std(pcc(lr2))/sqrt(length(lr2))...
      ,std(pcc(lr3))/sqrt(length(lr3))...
      ,std(pcc(lr4))/sqrt(length(lr4))...
      ,std(pcc(lr5))/sqrt(length(lr5))...
      ,std(pcc(lr6))/sqrt(length(lr6))];
subplot(1,3,3);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(lr1),pcc(lr2));
P(1,2) = p;
[~,p] = ttest2(pcc(lr1),pcc(lr3));
P(1,3) = p;
[~,p] = ttest2(pcc(lr1),pcc(lr4));
P(1,4) = p;
[~,p] = ttest2(pcc(lr2),pcc(lr3));
P(2,3) = p;
[~,p] = ttest2(pcc(lr2),pcc(lr4));
P(2,4) = p;
[~,p] = ttest2(pcc(lr3),pcc(lr4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(lrUnique(1)),num2str(lrUnique(2))...
            ,num2str(lrUnique(3)),num2str(lrUnique(4))...
            ,num2str(lrUnique(5)),num2str(lrUnique(6))});
ylabel('PCC');
title('Avg PCC Over All Sub and Sess','FontSize',13);
saveas(f,'result/lr.eps');

%% wd effect
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'WD Effect',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(wd1)),mean(ssim(wd2)),mean(ssim(wd3)),mean(ssim(wd4)),....
     mean(ssim(wd5)),mean(ssim(wd6)),mean(ssim(wd7)),mean(ssim(wd8))];
err = [std(ssim(wd1))/sqrt(length(wd1))...
      ,std(ssim(wd2))/sqrt(length(wd2))...
      ,std(ssim(wd3))/sqrt(length(wd3))...
      ,std(ssim(wd4))/sqrt(length(wd4))...
      ,std(ssim(wd5))/sqrt(length(wd5))...
      ,std(ssim(wd6))/sqrt(length(wd6))...
      ,std(ssim(wd7))/sqrt(length(wd7))...
      ,std(ssim(wd8))/sqrt(length(wd8))];
subplot(1,3,1);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(wd1),ssim(wd2));
P(1,2) = p;
[~,p] = ttest2(ssim(wd1),ssim(wd3));
P(1,3) = p;
[~,p] = ttest2(ssim(wd1),ssim(wd4));
P(1,4) = p;
[~,p] = ttest2(ssim(wd2),ssim(wd3));
P(2,3) = p;
[~,p] = ttest2(ssim(wd2),ssim(wd4));
P(2,4) = p;
[~,p] = ttest2(ssim(wd3),ssim(wd4));
P(3,4) = p;

% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(wdUnique(1)),num2str(wdUnique(2))...
            ,num2str(wdUnique(3)),num2str(wdUnique(4))...
            ,num2str(wdUnique(5)),num2str(wdUnique(6))...
            ,num2str(wdUnique(7)),num2str(wdUnique(8))});
title('Avg SSIM Over All Sub and Sessb','FontSize',13);
ylabel('SSIM');

y = [mean(mse(wd1)),mean(mse(wd2)),mean(mse(wd3)),mean(mse(wd4)),....
     mean(mse(wd5)),mean(mse(wd6)),mean(mse(wd7)),mean(mse(wd8))];
err = [std(mse(wd1))/sqrt(length(wd1))...
      ,std(mse(wd2))/sqrt(length(wd2))...
      ,std(mse(wd3))/sqrt(length(wd3))...
      ,std(mse(wd4))/sqrt(length(wd4))...
      ,std(mse(wd5))/sqrt(length(wd5))...
      ,std(mse(wd6))/sqrt(length(wd6))...
      ,std(mse(wd7))/sqrt(length(wd7))...
      ,std(mse(wd8))/sqrt(length(wd8))];
subplot(1,3,2);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(wd1),mse(wd2));
P(1,2) = p;
[~,p] = ttest2(mse(wd1),mse(wd3));
P(1,3) = p;
[~,p] = ttest2(mse(wd1),mse(wd4));
P(1,4) = p;
[~,p] = ttest2(mse(wd2),mse(wd3));
P(2,3) = p;
[~,p] = ttest2(mse(wd2),mse(wd4));
P(2,4) = p;
[~,p] = ttest2(mse(wd3),mse(wd4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(wdUnique(1)),num2str(wdUnique(2))...
            ,num2str(wdUnique(3)),num2str(wdUnique(4))...
            ,num2str(wdUnique(5)),num2str(wdUnique(6))...
            ,num2str(wdUnique(7)),num2str(wdUnique(8))});
ylabel('MSE');
title('Avg MSE Over All Sub and Sess','FontSize',13);

y = [mean(pcc(wd1)),mean(pcc(wd2)),mean(pcc(wd3)),mean(pcc(wd4)),....
     mean(pcc(wd5)),mean(pcc(wd6)),mean(pcc(wd7)),mean(pcc(wd8))];
err = [std(pcc(wd1))/sqrt(length(wd1))...
      ,std(pcc(wd2))/sqrt(length(wd2))...
      ,std(pcc(wd3))/sqrt(length(wd3))...
      ,std(pcc(wd4))/sqrt(length(wd4))...
      ,std(pcc(wd5))/sqrt(length(wd5))...
      ,std(pcc(wd6))/sqrt(length(wd6))...
      ,std(pcc(wd7))/sqrt(length(wd7))...
      ,std(pcc(wd8))/sqrt(length(wd8))];
subplot(1,3,3);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(wd1),pcc(wd2));
P(1,2) = p;
[~,p] = ttest2(pcc(wd1),pcc(wd3));
P(1,3) = p;
[~,p] = ttest2(pcc(wd1),pcc(wd4));
P(1,4) = p;
[~,p] = ttest2(pcc(wd2),pcc(wd3));
P(2,3) = p;
[~,p] = ttest2(pcc(wd2),pcc(wd4));
P(2,4) = p;
[~,p] = ttest2(pcc(wd3),pcc(wd4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(wdUnique(1)),num2str(wdUnique(2))...
            ,num2str(wdUnique(3)),num2str(wdUnique(4))...
            ,num2str(wdUnique(5)),num2str(wdUnique(6))...
            ,num2str(wdUnique(7)),num2str(wdUnique(8))});
ylabel('PCC');
title('Avg PCC Over All Sub and Sess','FontSize',13);
saveas(f,'result/wd.eps');

%% bs effect
f = figure('units','normalized','outerposition',[0 0 1 1]);
annotation('textbox',[.47 .9 .1 .1],'String',...
    'BS Effect',...
    'EdgeColor','none','FontSize',20);

y = [mean(ssim(bs1)),mean(ssim(bs2)),mean(ssim(bs3)),mean(ssim(bs4)),mean(ssim(bs5))];
err = [std(ssim(bs1))/sqrt(length(bs1))...
      ,std(ssim(bs2))/sqrt(length(bs2))...
      ,std(ssim(bs3))/sqrt(length(bs3))...
      ,std(ssim(bs4))/sqrt(length(bs4))...
      ,std(ssim(bs5))/sqrt(length(bs5))];
subplot(1,3,1);
P = nan(numel(y), numel(y));
[~,p] = ttest2(ssim(bs1),ssim(bs2));
P(1,2) = p;
[~,p] = ttest2(ssim(bs1),ssim(bs3));
P(1,3) = p;
[~,p] = ttest2(ssim(bs1),ssim(bs4));
P(1,4) = p;
[~,p] = ttest2(ssim(bs2),ssim(bs3));
P(2,3) = p;
[~,p] = ttest2(ssim(bs2),ssim(bs4));
P(2,4) = p;
[~,p] = ttest2(ssim(bs3),ssim(bs4));
P(3,4) = p;

% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(bsUnique(1)),num2str(bsUnique(2))...
            ,num2str(bsUnique(3)),num2str(bsUnique(4)),num2str(bsUnique(5))});
title('Avg SSIM Over All Sub and Sessb','FontSize',13);
ylabel('SSIM');

y = double([mean(mse(bs1)),mean(mse(bs2)),mean(mse(bs3)),mean(mse(bs4)),mean(mse(bs5))]);
err = [std(mse(bs1))/sqrt(length(bs1))...
      ,std(mse(bs2))/sqrt(length(bs2))...
      ,std(mse(bs3))/sqrt(length(bs3))...
      ,std(mse(bs4))/sqrt(length(bs4))...
      ,std(mse(bs5))/sqrt(length(bs5))];
subplot(1,3,2);
P = nan(numel(y), numel(y));
[~,p] = ttest2(mse(bs1),mse(bs2));
P(1,2) = p;
[~,p] = ttest2(mse(bs1),mse(bs3));
P(1,3) = p;
[~,p] = ttest2(mse(bs1),mse(bs4));
P(1,4) = p;
[~,p] = ttest2(mse(bs2),mse(bs3));
P(2,3) = p;
[~,p] = ttest2(mse(bs2),mse(bs4));
P(2,4) = p;
[~,p] = ttest2(mse(bs3),mse(bs4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(bsUnique(1)),num2str(bsUnique(2))...
            ,num2str(bsUnique(3)),num2str(bsUnique(4)),num2str(bsUnique(5))});
ylabel('MSE');
title('Avg MSE Over All Sub and Sess','FontSize',13);

y = [mean(pcc(bs1)),mean(pcc(bs2)),mean(pcc(bs3)),mean(pcc(bs4)),mean(pcc(bs5))];
err = [std(pcc(bs1))/sqrt(length(bs1))...
      ,std(pcc(bs2))/sqrt(length(bs2))...
      ,std(pcc(bs3))/sqrt(length(bs3))...
      ,std(pcc(bs4))/sqrt(length(bs4))...
      ,std(pcc(bs5))/sqrt(length(bs5))];
subplot(1,3,3);
P = nan(numel(y), numel(y));
[~,p] = ttest2(pcc(bs1),pcc(bs2));
P(1,2) = p;
[~,p] = ttest2(pcc(bs1),pcc(bs3));
P(1,3) = p;
[~,p] = ttest2(pcc(bs1),pcc(bs4));
P(1,4) = p;
[~,p] = ttest2(pcc(bs2),pcc(bs3));
P(2,3) = p;
[~,p] = ttest2(pcc(bs2),pcc(bs4));
P(2,4) = p;
[~,p] = ttest2(pcc(bs3),pcc(bs4));
P(3,4) = p;
% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);
b = superbar(1:length(y),y,'E',err,'ErrorbarStyle','T');
xticks(1:length(y));
xticklabels({num2str(bsUnique(1)),num2str(bsUnique(2))...
            ,num2str(bsUnique(3)),num2str(bsUnique(4)),num2str(bsUnique(5))});
ylabel('PCC');
title('Avg PCC Over All Sub and Sess','FontSize',13);
saveas(f,'result/bs.eps');
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

%%
target_char1 = intersect(sess2,char1);
target_char2 = intersect(sess2,char2);
target_char3 = intersect(sess2,char3);
target_char4 = intersect(sess2,char4);
target_char5 = intersect(sess2,char5);
target_char6 = intersect(sess2,char6);
target_char7 = intersect(sess2,char7);
target_char8 = intersect(sess2,char8);
target_good = intersect(sess2,good);
target_bad = intersect(sess2,bad);

eachbox = 8;
target_prop = ssim;
perc = 3;
xtick_var = 70:185:1375;
sort_type = 'descend';
crit = sub4;
saveName = 'result/ssim_sub4.eps';
titleName = 'Sess 2, Sub 4,  ( <----- Better SSIM) ';
target_char1 = intersect(target_char1,crit);
target_char2 = intersect(target_char2,crit);
target_char3 = intersect(target_char3,crit);
target_char4 = intersect(target_char4,crit);
target_char5 = intersect(target_char5,crit);
target_char6 = intersect(target_char6,crit);
target_char7 = intersect(target_char7,crit);
target_char8 = intersect(target_char8,crit);
target_good = intersect(target_good,crit);
target_bad = intersect(target_bad,crit);
%
f = figure('units','normalized','outerposition',[0 0 1 1]);

ssim_target = target_prop(target_char1);
original = {result.original};
original = original(target_char1);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char1);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,1);
mont = montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);
title('Good','FontSize',20,'Color','g');

ssim_target = target_prop(target_char2);
original = {result.original};
original = original(target_char2);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char2);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,3);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char3);
original = {result.original};
original = original(target_char3);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char3);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,5);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char4);
original = {result.original};
original = original(target_char4);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char4);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,7);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char5);
original = {result.original};
original = original(target_char5);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char5);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,2);
montage(temp','Size',[2,eachbox]);
title('Bad','FontSize',20,'Color','r');
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char6);
original = {result.original};
original = original(target_char6);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char6);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,4);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char7);
original = {result.original};
original = original(target_char7);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char7);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,6);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

ssim_target = target_prop(target_char8);
original = {result.original};
original = original(target_char8);
reconstruction = {result.reconstruction};
reconstruction = reconstruction(target_char8);
[~,ssim_sorted_index] = sort(ssim_target,sort_type);
temp = cell(2,eachbox);
ssim_str = cell(1,eachbox);
for i = 1:eachbox
    temp{1,i} = original{ssim_sorted_index(i)};
    temp{2,i} = reconstruction{ssim_sorted_index(i)};
    ssim_str{i} = num2str(ssim_target(ssim_sorted_index(i)),perc);
end
subplot_tight(4,2,8);
montage(temp','Size',[2,eachbox]);
axis on;
yticks([]);
xticks(xtick_var);
xticklabels(ssim_str);

t = suptitle(titleName);
t.FontSize = 25;
saveas(f,saveName,'epsc');
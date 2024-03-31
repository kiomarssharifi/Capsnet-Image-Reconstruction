clearvars;
clc;
sub = 6;
ses = 2;
rois = {'v1','v2','v3','v1.v2','v1.v2.v3'};
for iRoi = 1:length(rois)
    clearvars -except sub ses rois iRoi
    SUB = num2str(sub);
    SES = num2str(ses);
    roi = rois{iRoi};
    disp(roi);
    resultDir = '/media/hdd1/users/sharifi/capsnet/probe/';
    roiDir = ['/media/hdd1/users/sharifi/fmri/sub0',SUB,...
              '/sess0',SES,'/results.mni.noblur/'];

    deepIndex = dlmread(['./voxel_selection_index/deepIndex_0',...
                      SUB,'ses0',SES,'.txt']);
    M = dlmread([roiDir,'roi.data.',roi,'.sub0',SUB,'.sess0',SES,'.txt']);

    for i = 1:128
%         rest(i,:) = M(:,i*10-5)';
        probe(i,:) = M(:,int32(i*10))';
    end
%     restTemp = rest;
    probeTemp = probe;
    for i = 1:128
%         rest(deepIndex(i),:) = restTemp(i,:);
        probe(deepIndex(i),:) = probeTemp(i,:);
    end
    writematrix(probe,[resultDir,'probe.all.sub0',SUB,...
                '.ses0',SES,'.',roi,'.txt']);
    %%
    digitCaps = readmatrix('digitcaps.txt');
    digitCaps = reshape(digitCaps,[128,8,16]);
    [~,indx] = max(vecnorm(digitCaps,2,3),[],2);
    for i = 1:128
        digitCapsSquashed(i,:) = digitCaps(i,indx(i),:);
    end

    X = cat(2,ones(128,1),digitCapsSquashed);
    for iVoxel = 1:size(M,1)
        [~,~,~,~,stats] = regress(probe(:,iVoxel),X);
        R2(iVoxel) = stats(1);
        pVal(iVoxel) = stats(3);
    end
    [S,I] = sort(R2,'descend');
    writematrix(probe(:,I(1:128)),[resultDir,'probe.sub0',SUB,...
                '.ses0',SES,'.',roi,'.txt']);
    figure;
    plot(S,'LineWidth',3);
    title(['Sorted $R^2$ sub',SUB,' ses',SES,' roi ',roi],'interpreter','latex','FontSize',15);
    ylabel('$R^2$','interpreter','latex','FontSize',15);
    xlabel('Voxel','FontSize',15);
end
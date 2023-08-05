clc
clear
close all

N_sensor = 3;

for SensorNo = 1:N_sensor
    % SensorNo = 5;  % sensor No.
    SensorLoc = [10.102041,-3.42,-1.3;
        10.102041,-0.38,-1.3;
        10.102041,+3.42,-1.3;
        29.897959,-3.42,-1.3;
        29.897959,-0.38,-1.3;
        29.897959,+3.42,-1.3];
    % load NN identified influence surface
    ReadPath = 'C:\Users\xudjian\Desktop\Bridge Influence Surface Identification with MLP\';
    XX_NN = readmatrix([ReadPath,'X.csv']);
    YY_NN = readmatrix([ReadPath,'Y.csv']);
    tmp = readmatrix([ReadPath,'Z.csv']);
    Nx = 101;
    Ny = 101;
    for i = 1:N_sensor
        ZZ_NN{i} = tmp(1+(i-1)*Nx:i*Nx,:);
    end
    %% load FEM influence surface
    load IS_FEM.mat
    X_FEM = Y;  % here X and Y are reversed
    Y_FEM = X;
    Z_FEM = Z;
    clear X Y Z
    [XX_FEM,YY_FEM,ZZ_FEM]=griddata(X_FEM,Y_FEM,Z_FEM(:,SensorNo),XX_NN,YY_NN,'cubic');
    %% compare two influence surfaces - no meshgrid
%     close all
    figure
    surface1 = ZZ_NN{SensorNo}/10;
    surface2 = ZZ_FEM*10^6/10;

    % plot two surfaces in the same axis
    subplot(2,1,1)
    surf(XX_NN,YY_NN,surface1, 'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none')
    hold on
    surf(XX_FEM,YY_FEM,surface2, 'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none')
    scatter3(SensorLoc(SensorNo,2),SensorLoc(SensorNo,1),min(min(surface2)),'filled','red')   % here X and Y are reversed
    grid on
    daspect([1,1,1])
    legend('NN','FEM','sensor','location','best')
    % plot contourf of the error
    Error = surface2-surface1;
    view([45,30])
    subplot(2,1,2)
    contourf(XX_FEM,YY_FEM,Error)
    colorbar('northoutside')
    % colormap;
    hold on
    scatter3(SensorLoc(SensorNo,2),SensorLoc(SensorNo,1),10,'filled','red')
    daspect([1,1,1])
    view([90,90])
    legend('error','sensor')

    % calculate metrics of errors
    ErrorMean(SensorNo) = sum(sum(abs(Error)))/Nx/Ny;
    ErrorMax(SensorNo) = max(max(abs(Error)));
    ErrorStd(SensorNo) = std(std(abs(Error)));

    sgtitle(['ErrorMean = ',num2str(ErrorMean(SensorNo)),...
        '; ErrorMax = ',num2str(ErrorMax(SensorNo)),...
        '; ErrorStd = ',num2str(ErrorStd(SensorNo))])
end

output_index = [ErrorMean(3);ErrorMax(3);ErrorStd(3);
    ErrorMean(2);ErrorMax(2);ErrorStd(2);
    ErrorMean(1);ErrorMax(1);ErrorStd(1)];
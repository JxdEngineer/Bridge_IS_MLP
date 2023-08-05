%%
clc
clear
close all
load Quilligan_data_18kmh

tic
for SensorNo = 1:3
    % SensorNo = 6;  % sensor No.
    SensorLoc = [10.102041,-3.42,-1.3;
        10.102041,-0.38,-1.3;
        10.102041,+3.42,-1.3];
    %% identify influence surfaces using the Quilligan's method
    D = [0,6]; % distance between every axle the the first front axle (unit: m)
    A = [3.0696*2,3.2952*2];  % axle weight of every axle (from front to rear, unit: ton)
    AxleN = 2;
    fs = 200;
    for n = 1:length(u_raw)
        K=length(axle2_loc{n}(:,2));  % number of samplings
        v=(axle2_loc{n}(end,2)-axle2_loc{n}(1,2))/(K/fs); % average speed (unit: m/s)
        C=round(D*fs/v);
        W=zeros(K-C(AxleN),K-C(AxleN));
        % generate the upper part of W
        for i=1:K-C(AxleN)
            W(i,i)=A(1)^2+A(2)^2;
            if i+(C(2)-C(1))<=K-C(2)
                W(i,i+(C(2)-C(1)))=A(1)*A(2);
            end
        end
        % diagnalize W
        for i=1:K-C(AxleN)
            for j=1:K-C(AxleN)
                if i>j
                    W(i,j)=W(j,i);
                end
            end
        end
        % generate bridge response vector
        S=zeros(K-C(AxleN),1);
        for i=1:K-C(AxleN)
            S(i)=A(1)*u_raw{n}(i+C(1),SensorNo)+...
                A(2)*u_raw{n}(i+C(AxleN),SensorNo);
        end
        % calculate influence line
        Z_q1{n} = [W\S]';
        Z_q1{n} = Z_q1{n}-highpass(Z_q1{n},1,fs);   % remove the high frequency component
        Y_q1{n} = v*(1:K-C(AxleN))/fs;
        X_q1{n} = [axle2_loc{n}(end-length(Y_q1{n})+1:end,1)]';

        disp(n)
    end

    % use the IL points to fit a 3D surface
    X_q2 = cell2mat(X_q1);   % here X and Y are reversed
    Y_q2 = cell2mat(Y_q1);
    Z_q2 = cell2mat(Z_q1);

%     ReadPath = 'C:\Users\xudjian\Desktop\Bridge Influence Surface Identification with MLP\';
%     XX_NN = readmatrix([ReadPath,'X.csv']);
%     YY_NN = readmatrix([ReadPath,'Y.csv']);

    XX_NN = readmatrix('./X.csv');
    YY_NN = readmatrix('./Y.csv');

    [XX_q,YY_q,ZZ_q]=griddata(X_q2,Y_q2,Z_q2,XX_NN,YY_NN,'v4');
    %% load FEM influence surface
    load IS_FEM.mat
    X_FEM = Y;  % here X and Y are reversed
    Y_FEM = X;
    Z_FEM = Z;
    clear X Y Z
    [XX_FEM,YY_FEM,ZZ_FEM]=griddata(X_FEM,Y_FEM,Z_FEM(:,SensorNo),XX_NN,YY_NN,'cubic');
    %% compare two influence surfaces
%     close all
    figure
    % plot two surfaces in the same axis
    subplot(2,1,1)
    surf(XX_q,YY_q,ZZ_q/10*1000000, 'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none')
    hold on
    surf(XX_FEM,YY_FEM,ZZ_FEM*100000, 'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none')
    scatter3(SensorLoc(SensorNo,2),SensorLoc(SensorNo,1),min(min(ZZ_FEM*100000)),'filled','red')   % here X and Y are reversed
    grid on
    daspect([1,1,1])
    legend('Quilligan','FEM','sensor','location','best')
    % plot contourf of the error
    Error = ZZ_q/10*1000000-ZZ_FEM*100000;
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

    Nx = length(XX_FEM);
    Ny = length(YY_FEM);
    % calculate metrics of errors
    ErrorMean(SensorNo) = sum(sum(abs(Error)))/Nx/Ny;
    ErrorMax(SensorNo) = max(max(abs(Error)));
    ErrorStd(SensorNo) = std(std(abs(Error)));

    sgtitle(['ErrorMean = ',num2str(ErrorMean(SensorNo)),...
        '; ErrorMax = ',num2str(ErrorMax(SensorNo)),...
        '; ErrorStd = ',num2str(ErrorStd(SensorNo))])
end
toc
%% Structural Elements
filterDisk30 = strel('disk',3);
filterDisk0 = strel('disk',5);
filterDisk = strel('disk',25);
largeDisk = strel('disk',50);
filterDisk2 = strel('disk',7);
filterDisk3 = strel('disk',10);
numGLCMBins = 40;

%% Filename
quantiativeLesionData = [];
quantiativeControlData = [];
imagename = strcat(melanomapath,imageFile);

%% Image Loading
f=imread(imagename);
figure(1)
subplot(3,3,1)
imshow(f)
title(imageFile);
subplot(3,3,2)
imshow(f(:,:,2),[])
title("Green Channel");

Gmedian = median(double(f(:,:,2))/255.0,'all');


sz = size(f);
f=double(imresize(f,[512,512*sz(2)/sz(1)]))/255;
f = imclose(f,filterDisk0);
f = imopen(f,filterDisk0);
subplot(3,3,3)
imshow(f,[])
title("Shaved");
f = imcrop(f,[32 32 (512*sz(2)/sz(1)-64) 447 ]);
fcp = f;

RImage = f(:,:,1);
GImage = f(:,:,2);
BImage = f(:,:,3);

Gcenter = imcrop(GImage,[32 32 (448*sz(2)/sz(1)-64) 383 ]);

Ggain = 1.00/Gmedian;


f(:,:,1) = (RImage-GImage)*Ggain + 0.5;
f(:,:,2) = (GImage-Gmedian)*Ggain + 0.5;
f(:,:,3) = (BImage-RImage)*Ggain + 0.5;


fcenter = imcrop(fcp,[32 32 (448*sz(2)/sz(1)-64) 383 ]);

for channel = 1:3
    fcp(:,:,channel) = fcp(:,:,channel)-min(min(fcenter(:,:,channel)));
    fcp(:,:,channel) = fcp(:,:,channel)./max(max(fcp(:,:,channel)));
end

minImage = min(fcp,[],3);

figure(1)
subplot(3,3,4)
imshow(f,[])
title("Standardized");

maskthr = 0.75*(median(minImage,'all') + graythresh(minImage))/2.0;


mask = minImage <= maskthr;
subplot(3,3,5)
imshow(mask,[])
title('Raw Mask');
maskMorph = imdilate(mask,filterDisk3);
maskMorph = imclose(maskMorph,filterDisk3);
maskMorph = imopen(maskMorph,filterDisk);
finalmask = imclose(maskMorph,filterDisk);
%imshow(finalmask,[])
%title('Clean Mask');

finalmaskD = imdilate(maskMorph,filterDisk2);
controlmask = imdilate(finalmaskD,filterDisk); 
controlmask = imdilate(controlmask,filterDisk) & not(finalmaskD);

%%
subplot(3,3,6)
imshow(minImage,[0,max(max(minImage))])
title('Min minImage');
subplot(3,3,7)
lesionimage = f.*finalmask ;
imshow(lesionimage,[]);
title('Lesion Mask');

templateRadius =  sum(sum(finalmask))/3;
templateRadius2 = sum(sum(finalmask))/12;
sz = size(finalmask);
[X,Y] = meshgrid(1:sz(2),1:sz(1));
templateMx = 0.7*mean(X(finalmask)) + 0.3*sz(2)/2;
templateMy = 0.7*mean(Y(finalmask)) + 0.3*sz(1)/2;
templateMx2 = 0.5*mean(X(finalmask)) + 0.5*sz(2)/2;
templateMy2 = 0.5*mean(Y(finalmask)) + 0.5*sz(1)/2;
finalmask = finalmask & (((X - templateMx).^2 + (Y - templateMy).^2) < templateRadius);
finalmask = imopen(finalmask,filterDisk3);
finalmask = imclose(finalmask,filterDisk);
if (templateRadius2 < 900 ) 
    templateRadius2 = 900;
end
finalmask = finalmask | (((X - templateMx2).^2 + (Y - templateMy2).^2) < templateRadius2);

controlmask = controlmask & not(finalmask) & imdilate(imdilate(finalmask,largeDisk),largeDisk);

lesionimage = f.*finalmask ;
subplot(3,3,8)
imshow(lesionimage,[]);
title('Lesion Sample ROI');

controlimage = f.*controlmask;
subplot(3,3,9)
imshow(controlimage)
title('Control ROI');


%% Extract Lession features
dy = [-1 -2 -1; 0 0 0; 1 2 1]/8;
dx = dy';
dy2 = [-1 -2 -1; 0 0 0; 0 0 0; 0 0 0; 1 2 1]/16;
dx2 = dy2';
maskarea = sum(sum(finalmask));
for channel = 1:3
    
    LesionData = f(:,:,channel);
    minv = min(min(LesionData));
    adLesionData = LesionData - minv;
    
%    imshow(LesionData,[]);
    histoData = reshape(LesionData(finalmask),1,[]);
    temp_mean = mean(histoData);
    temp_m2 = moment(histoData,2);
    temp_m3 = moment(histoData,3);
    temp_m4 = moment(histoData,4);
    
    masked_m2 = sqrt(temp_m2);
    masked_m3 = power(abs(temp_m3),1/3).*sign(temp_m3);
    masked_m4 = power(temp_m4,1/4);
    dq = quantile(histoData,[0.01,0.05,0.25,0.5,0.75,0.95,0.99]);
    cov = log(10000.0*masked_m2 + 1.0);
    q90cov = log(10000.0*(dq(6)-dq(2)) + 1.0);
    if (abs(temp_mean) > 0)
        cov = log(10000.0*masked_m2/abs(temp_mean) + 1.0);
    end
    if (abs(dq(4)) > 0)
        q90cov = log(10000.0*(dq(6)-dq(2))/abs(dq(4)) + 1.00);
    end

    [N,edges] = histcounts(histoData,32);
    N = N/sum(N);
    N = N.*log(N);
    TF = isnan(N);
    N(TF) = 0;

    entropy = -sum(N);
    

    quantiativeLesionData = [quantiativeLesionData,temp_mean,masked_m2,masked_m3,masked_m4,entropy,cov,q90cov];
    
    
    volume = sum(sum(adLesionData(finalmask)));
    grad = (abs(conv2(LesionData,dx,'same')) + abs(conv2(LesionData,dy,'same')))/2;
    grad2 = (abs(conv2(LesionData,dx2,'same')) + abs(conv2(LesionData,dy2,'same')))/2;
    gradsum = sum(sum(grad(finalmask)));
    gradsum2 = sum(sum(grad2(finalmask)));
    surface = 128*gradsum + maskarea;
    compactness = 512*volume/surface^(3/2);
    quantiativeLesionData = [quantiativeLesionData,log(1+volume),log(1+surface),compactness,gradsum/volume,gradsum2/gradsum];

    %%%%%%%%%%%%%%%% Temperature GLCM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    glcm_1 = graycomatrix(LesionData(finalmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 2; -2 2; -2 0; -2 -2],'Symmetric',true);
    glcm_2 = graycomatrix(LesionData(finalmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 4; -3 3; -4 0; -3 -3],'Symmetric',true);
    glcm_3 = graycomatrix(LesionData(finalmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 8; -6 6; -8 0; -6 -6],'Symmetric',true);
    glcm_4 = graycomatrix(LesionData(finalmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 16; -11 11; -16 0; -11 -11],'Symmetric',true);
    glcm_1 = sum(glcm_1,3);
    glcm_2 = sum(glcm_2,3);
    glcm_3 = sum(glcm_3,3);
    glcm_4 = sum(glcm_4,3);
    GLCM_stats_1 = graycoprops(glcm_1);
    GLCM_stats_2 = graycoprops(glcm_2);
    GLCM_stats_3 = graycoprops(glcm_3);
    GLCM_stats_4 = graycoprops(glcm_4);

    quantiativeLesionData = [quantiativeLesionData,GLCM_stats_1.Contrast,GLCM_stats_1.Correlation,GLCM_stats_1.Energy,GLCM_stats_1.Homogeneity];
    quantiativeLesionData = [quantiativeLesionData,GLCM_stats_2.Contrast,GLCM_stats_2.Correlation,GLCM_stats_2.Energy,GLCM_stats_2.Homogeneity];
    quantiativeLesionData = [quantiativeLesionData,GLCM_stats_3.Contrast,GLCM_stats_3.Correlation,GLCM_stats_3.Energy,GLCM_stats_3.Homogeneity];
    quantiativeLesionData = [quantiativeLesionData,GLCM_stats_4.Contrast,GLCM_stats_4.Correlation,GLCM_stats_4.Energy,GLCM_stats_4.Homogeneity];


    GLCMSlope = [abs(log(1/2*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_2.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_2.Correlation)/(1.001-GLCM_stats_3.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_3.Correlation)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_3.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_2.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(8)];
    meanGLCMSlope = mean(GLCMSlope);

    quantiativeLesionData = [quantiativeLesionData,meanGLCMSlope];

    GLCMSlope = [abs(log(1/2*GLCM_stats_1.Contrast/GLCM_stats_2.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*GLCM_stats_2.Contrast/GLCM_stats_3.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*GLCM_stats_1.Contrast/GLCM_stats_3.Contrast))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*GLCM_stats_3.Contrast/GLCM_stats_4.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*GLCM_stats_2.Contrast/GLCM_stats_4.Contrast))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*GLCM_stats_1.Contrast/GLCM_stats_4.Contrast))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeLesionData = [quantiativeLesionData,meanGLCMSlope];


    GLCMSlope = [abs(log(2*GLCM_stats_1.Energy/GLCM_stats_2.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(2*GLCM_stats_2.Energy/GLCM_stats_3.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(4*GLCM_stats_1.Energy/GLCM_stats_3.Energy))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(2*GLCM_stats_3.Energy/GLCM_stats_4.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(4*GLCM_stats_2.Energy/GLCM_stats_4.Energy))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(8*GLCM_stats_1.Energy/GLCM_stats_4.Energy))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeLesionData = [quantiativeLesionData,meanGLCMSlope];


    GLCMSlope = [abs(log(1/2*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_2.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_2.Homogeneity)/(1.001-GLCM_stats_3.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_3.Homogeneity)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_3.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_2.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeLesionData = [quantiativeLesionData,meanGLCMSlope];


end

quantiativeLesionData = [quantiativeLesionData,Ggain,Gmedian];
allquantiativeLesionData = [allquantiativeLesionData;quantiativeLesionData];


%% Extract Control features
maskarea = sum(sum(controlmask));
for channel = 1:3
    
    ControlData = f(:,:,channel);
    minv = min(min(ControlData));
    aControlData = ControlData - minv;
%    imshow(ControlData,[]);
    histoData = reshape(ControlData(controlmask),1,[]);
    temp_mean = mean(histoData);
    temp_m2 = moment(histoData,2);
    temp_m3 = moment(histoData,3);
    temp_m4 = moment(histoData,4);
    
    masked_m2 = sqrt(moment(histoData,2));
    masked_m3 = moment(histoData,3);
    masked_m3 = power(abs(masked_m3),1/3).*sign(masked_m3);
    masked_m4 = power(moment(histoData,4),1/4);
    dq = quantile(histoData,[0.01,0.05,0.25,0.5,0.75,0.95,0.99]);
    cov=0;
    q90cov=0;
    if (abs(temp_mean) > 0)
        cov = log(10000.0*masked_m2/abs(temp_mean) + 1.0);
    end
    if (abs(dq(4)) > 0)
        q90cov = log(10000.0*(dq(6)-dq(2))/abs(dq(4)) + 1.0);
    end

    [N,edges] = histcounts(histoData,32);
    N = N/sum(N);
    N = N.*log(N);
    TF = isnan(N);
    N(TF) = 0;

    entropy = -sum(N);
    

    quantiativeControlData = [quantiativeControlData,temp_mean,masked_m2,masked_m3,masked_m4,entropy,cov,q90cov];

    volume = sum(sum(aControlData(controlmask)));
    grad = (abs(conv2(ControlData,dx,'same')) + abs(conv2(ControlData,dy,'same')))/2;
    grad2 = (abs(conv2(ControlData,dx2,'same')) + abs(conv2(ControlData,dy2,'same')))/2;
    gradsum = sum(sum(grad(controlmask)));
    gradsum2 = sum(sum(grad2(controlmask)));
    surface = 128*gradsum + maskarea;
    compactness = 512*volume/surface^(3/2);
    quantiativeControlData = [quantiativeControlData,log(1.0+volume+1),log(1.0+surface),compactness,gradsum/volume,gradsum2/gradsum];

    
    %%%%%%%%%%%%%%%%  GLCM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    glcm_1 = graycomatrix(ControlData(controlmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 2; -2 2; -2 0; -2 -2],'Symmetric',true);
    glcm_2 = graycomatrix(ControlData(controlmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 4; -3 3; -4 0; -3 -3],'Symmetric',true);
    glcm_3 = graycomatrix(ControlData(controlmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 8; -6 6; -8 0; -6 -6],'Symmetric',true);
    glcm_4 = graycomatrix(ControlData(controlmask),'NumLevels',numGLCMBins,'GrayLimits',[-1.0,1.50],'Offset',[0 16; -11 11; -16 0; -11 -11],'Symmetric',true);
    glcm_1 = sum(glcm_1,3);
    glcm_2 = sum(glcm_2,3);
    glcm_3 = sum(glcm_3,3);
    glcm_4 = sum(glcm_4,3);
    GLCM_stats_1 = graycoprops(glcm_1);
    GLCM_stats_2 = graycoprops(glcm_2);
    GLCM_stats_3 = graycoprops(glcm_3);
    GLCM_stats_4 = graycoprops(glcm_4);

    quantiativeControlData = [quantiativeControlData,GLCM_stats_1.Contrast,GLCM_stats_1.Correlation,GLCM_stats_1.Energy,GLCM_stats_1.Homogeneity];
    quantiativeControlData = [quantiativeControlData,GLCM_stats_2.Contrast,GLCM_stats_2.Correlation,GLCM_stats_2.Energy,GLCM_stats_2.Homogeneity];
    quantiativeControlData = [quantiativeControlData,GLCM_stats_3.Contrast,GLCM_stats_3.Correlation,GLCM_stats_3.Energy,GLCM_stats_3.Homogeneity];
    quantiativeControlData = [quantiativeControlData,GLCM_stats_4.Contrast,GLCM_stats_4.Correlation,GLCM_stats_4.Energy,GLCM_stats_4.Homogeneity];


    GLCMSlope = [abs(log(1/2*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_2.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_2.Correlation)/(1.001-GLCM_stats_3.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_3.Correlation)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_3.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_2.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*(1.001-GLCM_stats_1.Correlation)/(1.001-GLCM_stats_4.Correlation)))/log(8)];
    meanGLCMSlope = mean(GLCMSlope);

    quantiativeControlData = [quantiativeControlData,meanGLCMSlope];

    GLCMSlope = [abs(log(1/2*GLCM_stats_1.Contrast/GLCM_stats_2.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*GLCM_stats_2.Contrast/GLCM_stats_3.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*GLCM_stats_1.Contrast/GLCM_stats_3.Contrast))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*GLCM_stats_3.Contrast/GLCM_stats_4.Contrast))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*GLCM_stats_2.Contrast/GLCM_stats_4.Contrast))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*GLCM_stats_1.Contrast/GLCM_stats_4.Contrast))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeControlData = [quantiativeControlData,meanGLCMSlope];


    GLCMSlope = [abs(log(2*GLCM_stats_1.Energy/GLCM_stats_2.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(2*GLCM_stats_2.Energy/GLCM_stats_3.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(4*GLCM_stats_1.Energy/GLCM_stats_3.Energy))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(2*GLCM_stats_3.Energy/GLCM_stats_4.Energy))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(4*GLCM_stats_2.Energy/GLCM_stats_4.Energy))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(8*GLCM_stats_1.Energy/GLCM_stats_4.Energy))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeControlData = [quantiativeControlData,meanGLCMSlope];


    GLCMSlope = [abs(log(1/2*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_2.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_2.Homogeneity)/(1.001-GLCM_stats_3.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_3.Homogeneity)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/2*(1.001-GLCM_stats_3.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(2)];
    GLCMSlope = [GLCMSlope,abs(log(1/4*(1.001-GLCM_stats_2.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(4)];
    GLCMSlope = [GLCMSlope,abs(log(1/8*(1.001-GLCM_stats_1.Homogeneity)/(1.001-GLCM_stats_4.Homogeneity)))/log(8)];

    meanGLCMSlope = mean(GLCMSlope);

    quantiativeControlData = [quantiativeControlData,meanGLCMSlope];


end



allquantiativeControlData = [allquantiativeControlData;quantiativeControlData];

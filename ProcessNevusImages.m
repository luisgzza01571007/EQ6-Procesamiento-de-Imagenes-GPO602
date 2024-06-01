melanomapath = 'C:\Users\tibur\OneDrive\Documentos\MATLAB\DataMelanoma\test\nevus\';
allquantiativeLesionData = [];
allquantiativeControlData = [];

files = dir(fullfile(melanomapath, '*.jpg')); % lista de archivos .jpg en la carpeta

for i = 1:length(files)
    imageFile = files(i).name;
    loadAndProcessQuantitateImage;
    
end

writematrix(allquantiativeLesionData,'NevusLesionFeatures.csv') 
writematrix(allquantiativeControlData,'NevusControlFeatures.csv')
